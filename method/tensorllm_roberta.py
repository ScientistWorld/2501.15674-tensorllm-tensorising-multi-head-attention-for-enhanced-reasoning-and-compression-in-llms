#!/usr/bin/env python3
"""TensorLLM (Gu et al., 2025) reproduction for RoBERTa-base.

Paper procedure, per reasoning task:

1. Multi-head tensorisation (Sec. III-A): split W^Q, W^K, W^V and W^O^T of one
   transformer layer into per-head d_model x d_v sub-matrices, stack the four
   sub-matrices of each head into a 3D tensor and the heads into a 4D tensor
   W_all in R^{d_model x d_v x 4 x h}.
2. Tucker decomposition with shared factor matrices (Sec. III-B, Eq. 10-14):
   W_all ~= G_all x_1 U1 x_2 U2 x_3 U3 (x_4 I), i.e. Tucker/HOOI on modes
   1-3 with the head mode left uncompressed, so every head keeps its own core
   slice in a common subspace.  As in the paper (TensorLy ``partial_tucker``
   defaults) HOOI is initialised with HOSVD and iterated until the change of
   relative reconstruction error falls below 1e-8 (at most 100 iterations).
3. Layer-selective application (Sec. IV): the reconstruction replaces the
   MHA weights of a single layer.  Layer and multilinear ranks are chosen on
   the visible validation split (the first 20% of each dataset), best accuracy
   first, then lower loss.

The selected structure (layer, multi-head layout, ranks of the three shared
modes) is written to ``outputs/submissions/tensorllm_roberta/<task>/patch.json``;
the scorers run the same HOOI on the original weights, rebuild the layer and
count the parameters (shared factors + core).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.mlm_tasks import (  # noqa: E402
    D_MODEL,
    HEAD_DIM,
    MATRICES,
    N_HEADS,
    N_LAYERS,
    SUBMISSIONS_DIR,
    Roberta,
    Task,
    load_json,
    pick_device,
    save_json,
    task_names,
    write_patch,
)

METHOD_KEY = "tensorllm_roberta"
TUCKER_EQUATION = "abch,ma,db,tc->tmhd"  # core[R1,R2,R3,h], U1[d,R1], U2[dv,R2], U3[4,R3] -> [4, d, h, dv]
DEFAULT_R1 = (64, 128, 256, 384, 512)
DEFAULT_R2 = (16, 32, 48, 64)
DEFAULT_R3 = (2, 3, 4)
ORIGINAL_MHA_PARAMS = 4 * D_MODEL * D_MODEL


# --------------------------------------------------------------------------
# Multi-head tensorisation
# --------------------------------------------------------------------------


def tensorize(canonical: torch.Tensor) -> torch.Tensor:
    """[4, d_model, h*d_v] (Q, K, V, O^T; head-major columns) -> W_all [d_model, d_v, 4, h]."""
    return canonical.reshape(4, D_MODEL, N_HEADS, HEAD_DIM).permute(1, 3, 0, 2).contiguous()


def detensorize(w_all: torch.Tensor) -> torch.Tensor:
    return w_all.permute(2, 0, 3, 1).reshape(4, D_MODEL, N_HEADS * HEAD_DIM).contiguous()


# --------------------------------------------------------------------------
# Tucker decomposition with shared factor matrices (HOOI, head mode untouched)
# --------------------------------------------------------------------------


def unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    return tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """tensor x_mode matrix, with matrix of shape [J, I_mode]."""
    moved = tensor.movedim(mode, 0)
    shape = moved.shape
    out = matrix @ moved.reshape(shape[0], -1)
    return out.reshape((matrix.shape[0],) + tuple(shape[1:])).movedim(0, mode)


def leading_left_singular_vectors(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-``rank`` left singular vectors (via the Gram matrix; same subspace as a truncated SVD)."""
    _, vectors = torch.linalg.eigh(matrix @ matrix.T)
    return vectors[:, -rank:].flip(-1).contiguous()


def valid_ranks(ranks: tuple[int, int, int]) -> bool:
    sizes = (D_MODEL, HEAD_DIM, 4)
    r1, r2, r3 = ranks
    if any(r < 1 or r > s for r, s in zip(ranks, sizes)):
        return False
    # multilinear rank feasibility (each rank <= product of the others, incl. the h heads)
    return r1 <= r2 * r3 * N_HEADS and r2 <= r1 * r3 * N_HEADS and r3 <= r1 * r2 * N_HEADS


def compressed_params(ranks: tuple[int, int, int]) -> int:
    r1, r2, r3 = ranks
    return D_MODEL * r1 + HEAD_DIM * r2 + 4 * r3 + r1 * r2 * r3 * N_HEADS


def compression_ratio(ranks: tuple[int, int, int]) -> float:
    return ORIGINAL_MHA_PARAMS / compressed_params(ranks)


def shared_factor_tucker(w_all: torch.Tensor, ranks, n_iter_max: int = 100, tol: float = 1e-8):
    """HOOI (De Lathauwer et al., 2000) on modes 0-2 of W_all, HOSVD initialisation.

    Mirrors TensorLy ``partial_tucker(tensor, rank, modes=[0, 1, 2], init='svd',
    n_iter_max=100, tol=1e-8)``.  Returns (core [R1,R2,R3,h], [U1, U2, U3], info).
    """
    modes = (0, 1, 2)
    w = w_all.double()
    factors = [leading_left_singular_vectors(unfold(w, m), r) for m, r in zip(modes, ranks)]
    norm_sq = float((w * w).sum())
    errors: list[float] = []
    iterations = 0
    for iteration in range(n_iter_max):
        for index, mode in enumerate(modes):
            projected = w
            for other_index, other in enumerate(modes):
                if other_index != index:
                    projected = mode_dot(projected, factors[other_index].T, other)
            factors[index] = leading_left_singular_vectors(unfold(projected, mode), ranks[index])
        core = w
        for index, mode in enumerate(modes):
            core = mode_dot(core, factors[index].T, mode)
        errors.append(math.sqrt(abs(norm_sq - float((core * core).sum()))) / math.sqrt(norm_sq))
        iterations = iteration + 1
        if iteration > 1 and abs(errors[-2] - errors[-1]) < tol:
            break
    core = w
    for index, mode in enumerate(modes):
        core = mode_dot(core, factors[index].T, mode)
    return core, factors, {"hooi_iterations": iterations, "relative_error": errors[-1]}


def reconstruct_canonical(core: torch.Tensor, factors) -> torch.Tensor:
    """[4, d_model, h*d_v] reconstruction from the shared-factor Tucker factors."""
    value = torch.einsum(TUCKER_EQUATION, core, *factors)
    return value.reshape(4, D_MODEL, N_HEADS * HEAD_DIM)


def overrides_for(layer: int, canonical: torch.Tensor) -> dict:
    canonical = canonical.to(torch.float32)
    return {layer: {matrix: Roberta.to_module(matrix, canonical[i]).contiguous() for i, matrix in enumerate(MATRICES)}}


def decompose_layer(model: Roberta, layer: int, ranks, n_iter_max: int, tol: float):
    w_all = tensorize(model.canonical(layer))
    core, factors, info = shared_factor_tucker(w_all, ranks, n_iter_max=n_iter_max, tol=tol)
    return core, factors, info


def patch_for(layer: int, ranks) -> dict:
    """Submission block: multi-head layout of Q, K, V, O^T, shared factors on modes 0-2.

    The head mode (3) gets no rank, i.e. it stays uncompressed (Eq. 10).  The
    scorers run the same HOOI on the original weights and rebuild
    W_all' = G_all x_1 U1 x_2 U2 x_3 U3.
    """
    r1, r2, r3 = (int(r) for r in ranks)
    return {"layer": int(layer), "targets": list(MATRICES), "layout": "heads",
            "ranks": {"0": r1, "1": r2, "2": r3}}


# --------------------------------------------------------------------------
# Layer / rank selection on the visible validation split
# --------------------------------------------------------------------------


def rank_grid(r1s, r2s, r3s) -> list[tuple[int, int, int]]:
    grid = []
    for r1 in r1s:
        for r2 in r2s:
            for r3 in r3s:
                ranks = (r1, r2, r3)
                if valid_ranks(ranks) and compression_ratio(ranks) > 1.0:
                    grid.append(ranks)
    return grid


def condition_key(layer: int, ranks) -> str:
    return f"L{layer}_r{ranks[0]}-{ranks[1]}-{ranks[2]}"


def run_search(model, tasks: dict, layers, grid, log_path: Path, n_iter_max: int, tol: float) -> dict:
    log = load_json(log_path, {"conditions": {}})
    conditions = log["conditions"]
    total = len(layers) * len(grid)
    done = 0
    started = time.time()
    for layer in layers:
        for ranks in grid:
            key = condition_key(layer, ranks)
            entry = conditions.get(key, {})
            done += 1
            if all(name in entry for name in tasks):
                continue
            core, factors, info = decompose_layer(model, layer, ranks, n_iter_max, tol)
            overrides = overrides_for(layer, reconstruct_canonical(core, factors))
            entry.update(
                {
                    "layer": layer,
                    "ranks": list(ranks),
                    "compression_ratio": compression_ratio(ranks),
                    **info,
                }
            )
            for name, task in tasks.items():
                if name in entry:
                    continue
                result = task.evaluate(model, overrides)
                entry[name] = {"accuracy": result["accuracy"], "loss": result["loss"], "finite": result["finite"]}
            conditions[key] = entry
            save_json(log_path, log)
            summary = " ".join(f"{n[:5]}={entry[n]['accuracy']:.2f}" for n in tasks)
            print(f"[{done}/{total}] {key} CR={entry['compression_ratio']:.2f} it={info['hooi_iterations']} "
                  f"err={info['relative_error']:.4f} {summary} ({time.time() - started:.0f}s)", flush=True)
    return log


def select(log: dict, name: str, allowed: set[str] | None = None) -> dict:
    candidates = [
        entry for key, entry in log["conditions"].items()
        if name in entry and entry[name]["finite"] and (allowed is None or key in allowed)
    ]
    if not candidates:
        raise RuntimeError(f"no finite search result for {name}")
    return max(candidates, key=lambda e: (e[name]["accuracy"], -e[name]["loss"], e["compression_ratio"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=None, help="default: every task in data/tasks")
    parser.add_argument("--layers", default=",".join(str(i) for i in range(N_LAYERS)))
    parser.add_argument("--r1", default=",".join(map(str, DEFAULT_R1)))
    parser.add_argument("--r2", default=",".join(map(str, DEFAULT_R2)))
    parser.add_argument("--r3", default=",".join(map(str, DEFAULT_R3)))
    parser.add_argument("--n-iter-max", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--limit", type=int, default=None, help="debug: use a prefix of the validation split")
    parser.add_argument("--out-dir", default=str(SUBMISSIONS_DIR / METHOD_KEY))
    parser.add_argument("--log", default=str(ROOT / "outputs" / "tensorllm_search" / "search_log.json"))
    args = parser.parse_args()

    device = pick_device()
    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)
    model = Roberta(device)
    names = args.experiments or task_names()
    tasks = {name: Task(name, device, limit=args.limit) for name in names}
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    grid = rank_grid(*[[int(x) for x in s.split(",") if x.strip()] for s in (args.r1, args.r2, args.r3)])
    print(f"search: {len(layers)} layers x {len(grid)} rank tuples; tasks={names}", flush=True)
    log = run_search(model, tasks, layers, grid, Path(args.log), args.n_iter_max, args.tol)

    allowed = {condition_key(layer, ranks) for layer in layers for ranks in grid}
    out_dir = Path(args.out_dir)
    selection = {}
    for name in names:
        best = select(log, name, allowed)
        layer, ranks = best["layer"], tuple(best["ranks"])
        info = {"hooi_iterations": best["hooi_iterations"], "relative_error": best["relative_error"]}
        notes = {
            "method": "TensorLLM shared-factor Tucker (HOOI) on one layer's multi-head attention",
            "layer": layer,
            "ranks": list(ranks),
            "validation": best[name],
            **info,
        }
        write_patch(out_dir / name, [patch_for(layer, ranks)], notes)
        selection[name] = {"layer": layer, "ranks": list(ranks), "compression_ratio": compression_ratio(ranks),
                           "validation": best[name], **info}
        print(f"selected {name}: layer {layer} ranks {ranks} CR={compression_ratio(ranks):.2f} "
              f"val_acc={best[name]['accuracy']:.2f} val_loss={best[name]['loss']:.3f}", flush=True)
    save_json(out_dir / "selection.json", selection)


if __name__ == "__main__":
    main()
