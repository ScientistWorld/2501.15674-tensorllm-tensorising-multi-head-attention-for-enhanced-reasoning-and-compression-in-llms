#!/usr/bin/env python3
"""Round-2 self-test of the scorer changes (runs inside the job, before anything else).

1. Every structure patch that the round-1 scorer accepted must be materialised
   bit-identically by the round-2 scorer (same replaced matrices, same parameter
   count): the generalised layouts / head subsets must not move any existing
   score.  The round-1 scorer is kept verbatim in ``legacy/eval_test_round1.py``.
2. Explicit spellings of the named layouts and of "all heads" give identical
   weights.
3. Head-subset blocks are counted per (layer, matrix, head) slice, overlaps and
   malformed layouts are rejected.
4. The gate statistics behave as documented.

Exit status 1 on any failure (the job stops there).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SUBS = ROOT / "outputs" / "hardening" / "submissions"
TASKS = ("bios_profession_roberta", "fever_roberta_binary")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW_TEST = load("scorer_test_round2", ROOT / "eval" / "test" / "evaluate.py")
NEW_TRAIN = load("scorer_train_round2", ROOT / "eval" / "train" / "evaluate.py")
OLD = load("scorer_round1", ROOT / "redteam" / "hardening" / "legacy" / "eval_test_round1.py")

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def write(tmp: Path, name: str, blocks) -> Path:
    path = tmp / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "patch.json").write_text(json.dumps({"format": NEW_TEST.PATCH_FORMAT, "blocks": blocks}))
    return path


def max_diff(a, b) -> float:
    """Largest absolute difference between two override sets (inf if they replace different matrices)."""
    if set(a) != set(b):
        return math.inf
    return max((float((a[k] - b[k]).abs().max()) for k in a), default=0.0)


def same_overrides(a, b, tol: float = 1e-6) -> bool:
    diff = max_diff(a, b)
    if diff != 0.0:
        print(f"     (max |difference| {diff:.3e})", flush=True)
    return diff <= tol


def main() -> int:
    device = NEW_TEST.configure_torch()
    model = NEW_TEST.FixedRoberta(device)

    # 1. round-1 patches: identical materialisation in both round-2 scorers
    patch_dirs = sorted(p.parent for p in SUBS.glob("*/*/patch.json"))
    patch_dirs += sorted(p.parent for p in (ROOT / "outputs" / "submissions").glob("*/*/patch.json"))
    n_compared = 0
    for patch_dir in patch_dirs:
        try:
            old_over, old_info = OLD.materialize_patch(patch_dir, model)
        except OLD.PatchError as exc:
            for module in (NEW_TEST, NEW_TRAIN):
                try:
                    module.materialize_patch(patch_dir, model)
                    check(False, f"{patch_dir.relative_to(ROOT)} rejected by round 1 ({exc}) but accepted now")
                except module.PatchError:
                    pass
            continue
        for module in (NEW_TEST, NEW_TRAIN):
            new_over, new_info = module.materialize_patch(patch_dir, model)
            check(same_overrides(old_over, new_over)
                  and old_info["compression_ratio"] == new_info["compression_ratio"]
                  and old_info["stored_params"] == new_info["stored_params"],
                  f"{module.__name__}: {patch_dir.relative_to(ROOT)} identical "
                  f"(CR {new_info['compression_ratio']:.4f})")
        n_compared += 1
    check(n_compared >= 8, f"compared {n_compared} accepted round-1 patches")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        all_heads = list(range(12))
        mats = ["query", "key", "value", "output"]
        ref = write(tmp, "ref", [{"layer": 7, "targets": mats, "layout": "heads", "ranks": {"0": 64, "1": 48, "2": 4}}])
        explicit = write(tmp, "explicit", [{"layer": 7, "targets": mats, "heads": all_heads,
                                            "layout": [["m"], ["d"], ["t"], ["h"]],
                                            "ranks": {"0": 64, "1": 48, "2": 4}}])
        a, ia = NEW_TEST.materialize_patch(ref, model)
        b, ib = NEW_TEST.materialize_patch(explicit, model)
        check(same_overrides(a, b) and ia["compression_ratio"] == ib["compression_ratio"],
              "explicit heads layout + all heads == named 'heads' layout")
        stack = write(tmp, "stack", [{"layer": 2, "targets": ["value"], "layout": "stack", "ranks": {"0": 96}}])
        stack_x = write(tmp, "stack_x", [{"layer": 2, "targets": ["value"], "layout": [["m"], ["h", "d"], ["t"]],
                                          "ranks": {"0": 96}}])
        a, ia = NEW_TEST.materialize_patch(stack, model)
        b, ib = NEW_TEST.materialize_patch(stack_x, model)
        check(same_overrides(a, b) and ia["compression_ratio"] == ib["compression_ratio"],
              "explicit stack layout == named 'stack' layout")

        # a single-matrix stack block equals a truncated SVD of that matrix
        w = model.canonical_matrix(2, "value").double()
        u, s, vt = torch.linalg.svd(w, full_matrices=False)
        svd = ((u[:, :96] * s[:96]) @ vt[:96]).float()
        rel = float((a[(2, "value")] - svd).norm() / svd.norm())
        print(f"info stack block of one matrix vs truncated SVD: relative difference {rel:.2e}", flush=True)

        # head subsets: counting per (layer, matrix, head) slice
        half = write(tmp, "half", [{"layer": 3, "targets": ["query"], "heads": [0, 1, 2, 3, 4, 5],
                                    "layout": "stack", "ranks": {"0": 32}}])
        over, info = NEW_TEST.materialize_patch(half, model)
        stored = 768 * 32 + 32 * 384 * 1
        represented = stored + 768 * 64 * (48 - 6)
        check(info["stored_params"] == stored and info["represented_params"] == represented
              and abs(info["compression_ratio"] - 4 * 768 * 768 / represented) < 1e-12,
              f"head-subset parameter count ({info['represented_params']} == {represented})")
        orig = model.canonical_matrix(3, "query").reshape(768, 12, 64)
        new = over[(3, "query")].reshape(768, 12, 64)
        check(torch.equal(new[:, 6:, :], orig[:, 6:, :]) and not torch.equal(new[:, :6, :], orig[:, :6, :]),
              "head-subset block replaces only its heads")
        groups = write(tmp, "groups", [
            {"layer": 7, "targets": mats, "heads": [0, 2, 4, 6, 8, 10], "layout": "heads",
             "ranks": {"0": 64, "1": 48, "2": 4}},
            {"layer": 7, "targets": mats, "heads": [1, 3, 5, 7, 9, 11], "layout": "heads",
             "ranks": {"0": 64, "1": 48, "2": 4}},
        ])
        over, info = NEW_TEST.materialize_patch(groups, model)
        expected = 2 * (768 * 64 + 64 * 48 + 4 * 4 + 64 * 48 * 4 * 6)
        check(info["represented_params"] == expected and len(over) == 4,
              f"two head groups counted as two blocks ({info['represented_params']} == {expected})")

        custom = write(tmp, "custom", [{"layer": 5, "targets": mats, "layout": [["m", "t"], ["h", "d"]],
                                        "ranks": {"0": 256}}])
        over, info = NEW_TEST.materialize_patch(custom, model)
        expected = 3072 * 256 + 256 * 768
        check(info["represented_params"] == expected and all(bool(torch.isfinite(v).all()) for v in over.values()),
              f"custom 2-mode layout counted ({info['represented_params']} == {expected})")

        bad = {
            "overlap": [{"layer": 3, "targets": ["query"], "heads": [0, 1], "layout": "stack", "ranks": {"0": 8}},
                        {"layer": 3, "targets": ["query", "key"], "heads": [1, 2], "layout": "stack",
                         "ranks": {"0": 8}}],
            "missing_axis": [{"layer": 3, "targets": mats, "layout": [["m"], ["d"], ["t"]], "ranks": {"0": 8}}],
            "repeated_axis": [{"layer": 3, "targets": mats, "layout": [["m"], ["d"], ["t", "h", "m"]],
                               "ranks": {"0": 8}}],
            "unknown_axis": [{"layer": 3, "targets": mats, "layout": [["m"], ["d"], ["t"], ["x"]],
                              "ranks": {"0": 8}}],
            "bad_heads": [{"layer": 3, "targets": mats, "heads": [12], "layout": "heads", "ranks": {"0": 8}}],
            "dup_heads": [{"layer": 3, "targets": mats, "heads": [1, 1], "layout": "heads", "ranks": {"0": 8}}],
            "empty_heads": [{"layer": 3, "targets": mats, "heads": [], "layout": "heads", "ranks": {"0": 8}}],
            "bool_head": [{"layer": 3, "targets": mats, "heads": [True], "layout": "heads", "ranks": {"0": 8}}],
            "not_smaller": [{"layer": 4, "targets": mats, "heads": [h], "layout": "heads", "ranks": {"0": 256}}
                            for h in range(12)],
            "mode_out_of_range": [{"layer": 3, "targets": mats, "layout": [["m", "t"], ["h", "d"]],
                                   "ranks": {"2": 4}}],
        }
        for name, blocks in bad.items():
            path = write(tmp, f"bad_{name}", blocks)
            for module in (NEW_TEST, NEW_TRAIN):
                try:
                    module.materialize_patch(path, model)
                    check(False, f"{module.__name__}: invalid patch '{name}' rejected")
                except module.PatchError as exc:
                    check(True, f"{module.__name__}: invalid patch '{name}' rejected ({exc})")

    # 4. gate statistics
    z_p, z_m = NEW_TEST.gate_statistics([1, 1, 1, 0], [0, 0, 1, 0], 0.5)
    check(abs(z_p - 2 / math.sqrt(2)) < 1e-12 and abs(z_m - (0.75 - 0.5) / math.sqrt(0.25 / 4)) < 1e-12,
          f"gate statistics ({z_p:.4f}, {z_m:.4f})")
    z_p, _ = NEW_TEST.gate_statistics([1, 0, 1], [1, 0, 1], 0.5)
    check(z_p == 0.0, "identical predictions give z = 0 (gate fails)")

    print(f"\nself-test: {len(FAILURES)} failure(s)", flush=True)
    for failure in FAILURES:
        print("  FAIL", failure, flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
