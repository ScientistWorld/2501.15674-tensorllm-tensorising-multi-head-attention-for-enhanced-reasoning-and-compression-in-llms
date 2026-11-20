#!/usr/bin/env python3
"""Run a submitted data-free decomposition with no access to datasets or answers.

This is a CPU child, with Linux Landlock filesystem confinement. Only its
scratch directory and system libraries are readable; the workspace is denied.
No elapsed-time cutoff is used. The parent independently validates and contracts
its safetensors tensor network. See kernel.org/doc/html/latest/userspace-api/landlock.html.
"""
import ctypes
import json
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file


def confine(scratch):
    libc = ctypes.CDLL(None, use_errno=True)
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Landlock ABI >= 3 is required for submitted procedures")
    handled = (1 << 15) - 1  # filesystem rights through TRUNCATE (ABI 3)
    rules = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    def allow(path, rights):
        if not Path(path).exists():
            return
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = Beneath(rights, pfd)
        rc = libc.syscall(445, fd, 1, ctypes.byref(rule), 0)
        os.close(pfd)
        if rc:
            raise OSError(ctypes.get_errno(), "landlock_add_rule")
    # EXECUTE, READ_FILE, READ_DIR. No writable system paths.
    for path in ("/usr", "/opt", "/lib", "/lib64"):
        allow(path, 1 | 4 | 8)
    for path in ("/dev/null", "/dev/urandom"):
        allow(path, 2 | 4)
    allow(scratch, handled)
    if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(fd)


def main():
    scratch = Path(sys.argv[1]).resolve()
    os.chdir(scratch)
    # Read only the copied source/config and original weight tensor before confinement.
    source = (scratch / "procedure.py").read_text()
    params = json.loads((scratch / "params.json").read_text())
    original = load_file(str(scratch / "input.safetensors"))["original"]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.random.default_generator.manual_seed(0)
    confine(str(scratch))
    scope = {"__name__": "submitted_decomposition", "__file__": str(scratch / "procedure.py")}
    exec(compile(source, scope["__file__"], "exec"), scope)
    # Arbitrary data-free algorithm; returns an explicit tensor network.
    equation, tensors = scope["decompose"](original, params)
    if not isinstance(equation, str) or not isinstance(tensors, (tuple, list)):
        raise ValueError("decompose must return (einsum equation, list of tensors)")
    tensors = {f"f{i}": t.detach().cpu().double().contiguous() for i, t in enumerate(tensors)}
    save_file(tensors, str(scratch / "network.safetensors"))
    (scratch / "network.json").write_text(json.dumps({"equation": equation, "n_tensors": len(tensors)}))

if __name__ == "__main__":
    main()
