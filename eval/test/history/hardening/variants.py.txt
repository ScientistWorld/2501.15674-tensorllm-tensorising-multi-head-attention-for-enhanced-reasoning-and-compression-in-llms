#!/usr/bin/env python3
"""Hardening audit: build better/weaker variants and shortcut submissions.

Everything is written under ``outputs/hardening/submissions/<name>/<task>/`` in
the same structure-patch format a future method must use, then scored by the
unchanged scorers (``eval/train/evaluate.py`` and ``eval/test/evaluate.py``)
with ``--submissions outputs/hardening/submissions``.  Selection inside every
variant uses only the visible validation split.

    python redteam/hardening/variants.py            # all variants
    python redteam/hardening/variants.py --only better_greedy_layers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.mlm_tasks import (  # noqa: E402
    MATRICES,
    N_LAYERS,
    Roberta,
    Task,
    load_json,
    pick_device,
    save_json,
    task_names,
    write_patch,
)
from method.tensorllm_roberta import (  # noqa: E402
    decompose_layer,
    overrides_for,
    patch_for,
    reconstruct_canonical,
)

OUT = ROOT / "outputs" / "hardening"
SUBS = OUT / "submissions"
REFERENCE_DIR = ROOT / "outputs" / "submissions" / "tensorllm_roberta"
SEARCH_LOG = ROOT / "outputs" / "tensorllm_search" / "search_log.json"
LASER_RANKS = (8, 38, 77, 154, 307)  # ~1%, 5%, 10%, 20%, 40% of 768 singular values kept


def merge(*override_dicts):
    merged = {}
    for overrides in override_dicts:
        for layer, mats in overrides.items():
            merged.setdefault(layer, {}).update(mats)
    return merged


class Ctx:
    def __init__(self, n_iter_max: int, tol: float):
        self.device = pick_device()
        self.model = Roberta(self.device)
        self.names = task_names()
        self.tasks = {name: Task(name, self.device) for name in self.names}
        self.selection = json.loads((REFERENCE_DIR / "selection.json").read_text())
        self.log = load_json(SEARCH_LOG, {"conditions": {}})
        self.n_iter_max = n_iter_max
        self.tol = tol
        self._recon: dict = {}

    def recon(self, layer: int, ranks) -> torch.Tensor:
        key = (layer, tuple(ranks))
        if key not in self._recon:
            core, factors, _ = decompose_layer(self.model, layer, tuple(ranks), self.n_iter_max, self.tol)
            self._recon[key] = reconstruct_canonical(core, factors).float()
            if len(self._recon) > 160:
                self._recon.pop(next(iter(self._recon)))
        return self._recon[key]


def write_tucker(name: str, task: str, configs, notes=None) -> None:
    write_patch(SUBS / name / task, [patch_for(layer, ranks) for layer, ranks in configs], notes)


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


def previous_reference(ctx: Ctx) -> None:
    """The earlier gym's reference setting (layer 6, ranks (384,48,4)) for every task."""
    for task in ctx.names:
        write_tucker("previous_reference", task, [(6, (384, 48, 4))],
                     notes={"variant": "previous gym reference configuration"})


def weaker_low_rank(ctx: Ctx) -> None:
    """Reference layer, but the most aggressive ranks of the grid (64,16,2) -> CR 31.6."""
    for task in ctx.names:
        layer = ctx.selection[task]["layer"]
        write_tucker("weaker_low_rank", task, [(layer, (64, 16, 2))], notes={"variant": "low ranks"})


def shortcut_collapse_compress(ctx: Ctx) -> None:
    """Rank-(1,1,1) Tucker of the last layer: ~2782x compression, reasoning destroyed."""
    for task in ctx.names:
        write_tucker("shortcut_collapse_compress", task, [(11, (1, 1, 1))], notes={"variant": "collapse"})


def weaker_laser_mha(ctx: Ctx) -> None:
    """Key idea removed: LASER-style truncated SVD of ONE attention matrix, no multi-head tensor or sharing."""
    log_path = OUT / "laser_mha_log.json"
    log = load_json(log_path, {})
    for layer in range(N_LAYERS):
        canonical = ctx.model.canonical(layer).double()
        for m_index, matrix in enumerate(MATRICES):
            u, s, vt = torch.linalg.svd(canonical[m_index], full_matrices=False)
            for rank in LASER_RANKS:
                key = f"L{layer}_{matrix}_k{rank}"
                if key in log and all(t in log[key] for t in ctx.names):
                    continue
                approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
                override = {layer: {matrix: Roberta.to_module(matrix, approx.float()).contiguous()}}
                entry = log.setdefault(key, {"layer": layer, "matrix": matrix, "rank": rank})
                for name, task in ctx.tasks.items():
                    res = task.evaluate(ctx.model, override)
                    entry[name] = {"accuracy": res["accuracy"], "loss": res["loss"], "finite": res["finite"]}
                save_json(log_path, log)
    for name in ctx.names:
        best = max((e for e in log.values() if e[name]["finite"]),
                   key=lambda e: (e[name]["accuracy"], -e[name]["loss"]))
        layer, matrix, rank = best["layer"], best["matrix"], best["rank"]
        # rank-k truncated SVD of one matrix == "stack" layout of that matrix, rank k on the model mode
        block = {"layer": layer, "targets": [matrix], "layout": "stack", "ranks": {"0": rank}}
        write_patch(SUBS / "weaker_laser_mha" / name, [block],
                    {"variant": "LASER on one MHA matrix", "validation": best[name]})
        print(f"weaker_laser_mha {name}: layer {layer} {matrix} k={rank} val={best[name]}", flush=True)


def _greedy(ctx: Ctx, name: str, task, min_gain: float, top_k: int, max_layers: int):
    sel = ctx.selection[name]
    configs = [(sel["layer"], tuple(sel["ranks"]))]
    current = sel["validation"]
    per_layer: dict[int, list] = {}
    for entry in ctx.log["conditions"].values():
        if name in entry and entry[name]["finite"]:
            per_layer.setdefault(entry["layer"], []).append(entry)
    history = [{"configs": [[c[0], list(c[1])] for c in configs], "validation": current}]
    while len(configs) < max_layers:
        used = {layer for layer, _ in configs}
        base = merge(*[overrides_for(layer, ctx.recon(layer, ranks)) for layer, ranks in configs])
        best = None
        for layer in range(N_LAYERS):
            if layer in used:
                continue
            ranked = sorted(per_layer.get(layer, []), key=lambda e: (-e[name]["accuracy"], e[name]["loss"]))
            for entry in ranked[:top_k]:
                ranks = tuple(entry["ranks"])
                res = task.evaluate(ctx.model, merge(base, overrides_for(layer, ctx.recon(layer, ranks))))
                if not res["finite"]:
                    continue
                if best is None or (res["accuracy"], -res["loss"]) > (best[1]["accuracy"], -best[1]["loss"]):
                    best = ((layer, ranks), {"accuracy": res["accuracy"], "loss": res["loss"]})
        history.append({"candidate": [best[0][0], list(best[0][1])] if best else None,
                        "validation": best[1] if best else None})
        if best is None or best[1]["accuracy"] < current["accuracy"] + min_gain:
            break
        configs.append(best[0])
        current = best[1]
        print(f"greedy(min_gain={min_gain}) {name}: {configs} val={current}", flush=True)
    return configs, current, history


def better_greedy_layers(ctx: Ctx, min_gain: float = 1.0, top_k: int = 4, max_layers: int = 3) -> None:
    """Multi-layer TensorLLM: greedily add a further layer while it raises visible-split accuracy by
    at least ``min_gain`` points (about two standard errors of the visible split)."""
    state_path = OUT / "greedy_layers_margin.json"
    state = load_json(state_path, {})
    for name, task in ctx.tasks.items():
        if not state.get(name, {}).get("done"):
            configs, current, history = _greedy(ctx, name, task, min_gain, top_k, max_layers)
            state[name] = {"done": True, "configs": [[c[0], list(c[1])] for c in configs],
                           "validation": current, "history": history}
            save_json(state_path, state)
        configs = [(layer, tuple(ranks)) for layer, ranks in state[name]["configs"]]
        write_tucker("better_greedy_layers", name, configs,
                     notes={"variant": f"greedy multi-layer TensorLLM, min gain {min_gain}",
                            "validation": state[name]["validation"]})


def greedy_layers_nomargin(ctx: Ctx) -> None:
    """Same greedy, accepting any strict visible-split gain (selection overfits small gains)."""
    state_path = OUT / "greedy_layers.json"
    state = load_json(state_path, {})
    for name, task in ctx.tasks.items():
        if not state.get(name, {}).get("done"):
            configs, current, history = _greedy(ctx, name, task, 1e-9, 4, 3)
            state[name] = {"done": True, "configs": [[c[0], list(c[1])] for c in configs],
                           "validation": current, "history": history}
            save_json(state_path, state)
        configs = [(layer, tuple(ranks)) for layer, ranks in state[name]["configs"]]
        write_tucker("greedy_layers_nomargin", name, configs,
                     notes={"variant": "greedy multi-layer TensorLLM, any gain",
                            "validation": state[name]["validation"]})


def artifact_shortcuts(ctx: Ctx) -> None:
    """Patches that try to game the scorer's bookkeeping rather than the model."""
    from safetensors.torch import save_file

    ref = REFERENCE_DIR
    gen = torch.Generator().manual_seed(0)
    for name in ctx.names:
        layer = ctx.selection[name]["layer"]
        ranks = ctx.selection[name]["ranks"]
        # 1. reference structure + forged metrics + submitted factor/core tensors: all ignored
        dst = SUBS / "shortcut_forged_metadata" / name
        dst.mkdir(parents=True, exist_ok=True)
        spec = json.loads((ref / name / "patch.json").read_text())
        spec.update({"compression_ratio": 1e6, "accuracy": 100.0, "loss": 0.0, "stored_params": 1})
        spec["notes"] = {"accuracy": 100.0, "compression_ratio": 1e6, "verdict": "pass"}
        spec["blocks"][0]["factors"] = {"0": "U_model"}
        spec["blocks"][0]["core"] = "core"
        (dst / "patch.json").write_text(json.dumps(spec, indent=2))
        save_file({"U_model": torch.randn(768, ranks[0], generator=gen),
                   "core": torch.randn(ranks[0], ranks[1], ranks[2], 12, generator=gen)},
                  str(dst / "tensors.safetensors"))

        # 2. a factor-carrying patch in the earlier projection format (trained subspaces): rejected
        dst = SUBS / "shortcut_trained_factors" / name
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "patch.json").write_text(json.dumps({
            "format": "mha-projection-patch-v2",
            "blocks": [{"layer": layer, "targets": list(MATRICES), "layout": "heads", "factors": {"0": "U_model"}}],
        }))
        save_file({"U_model": torch.randn(768, ranks[0], generator=gen)}, str(dst / "tensors.safetensors"))

        # 3. "dense replacement": full ranks on every mode (no compression)
        write_patch(SUBS / "shortcut_dense_replacement" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads",
                      "ranks": {"0": 768, "1": 64, "2": 4}}])
        # 4. multilinear ranks that cannot exist (R1 > R2 * R3 * h)
        write_patch(SUBS / "shortcut_infeasible_ranks" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads",
                      "ranks": {"0": 512, "1": 16, "2": 2}}])
        # 5. rank 0 / unknown layout
        write_patch(SUBS / "shortcut_invalid_rank" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads", "ranks": {"0": 0}}])
        write_patch(SUBS / "shortcut_bad_layout" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "rows", "ranks": {"0": 64}}])

    # 6. only one task submitted
    dst = SUBS / "shortcut_missing_experiment" / "fever_roberta_binary"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ref / "fever_roberta_binary", dst, dirs_exist_ok=True)
    # 7. empty submission directory
    (SUBS / "shortcut_empty").mkdir(parents=True, exist_ok=True)
    # 8. a directory that claims the evaluator-owned baseline key (holds a collapse patch)
    for name in ctx.names:
        write_tucker("roberta_original", name, [(11, (1, 1, 1))])


def copy_reference() -> None:
    dst = SUBS / "tensorllm_roberta"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(REFERENCE_DIR, dst)


VARIANTS = {
    "previous_reference": previous_reference,
    "weaker_low_rank": weaker_low_rank,
    "weaker_laser_mha": weaker_laser_mha,
    "better_greedy_layers": better_greedy_layers,
    "greedy_layers_nomargin": greedy_layers_nomargin,
    "shortcut_collapse_compress": shortcut_collapse_compress,
    "artifact_shortcuts": artifact_shortcuts,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--n-iter-max", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()
    SUBS.mkdir(parents=True, exist_ok=True)
    copy_reference()
    ctx = Ctx(args.n_iter_max, args.tol)
    done_path = OUT / "variants_done.json"
    done = load_json(done_path, {})
    for name, fn in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        if done.get(name):
            print(f"skip {name} (done)", flush=True)
            continue
        started = time.time()
        print(f"=== {name}", flush=True)
        fn(ctx)
        done[name] = round(time.time() - started, 1)
        save_json(done_path, done)


if __name__ == "__main__":
    main()
