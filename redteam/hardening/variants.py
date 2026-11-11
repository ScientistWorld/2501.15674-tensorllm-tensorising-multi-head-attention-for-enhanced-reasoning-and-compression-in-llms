#!/usr/bin/env python3
"""Hardening audit: build better/weaker variants and shortcut submissions.

Everything is written under ``outputs/hardening/submissions/<name>/<task>/`` in
the same factor-patch format a future method must use, then scored by the
unchanged scorers (``eval/train/evaluate.py`` and ``eval/test/evaluate.py``)
with ``--submissions outputs/hardening/submissions``.  Selection inside every
variant uses only the visible validation split.

    python redteam/hardening/variants.py            # all variants
    python redteam/hardening/variants.py --only better_greedy_layers
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.mlm_tasks import (  # noqa: E402
    D_MODEL,
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
    TUCKER_EQUATION,
    compression_ratio,
    condition_key,
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


def write_tucker(name: str, task: str, configs, ctx: Ctx, n_iter_max=None, notes=None) -> None:
    blocks, tensors = [], {}
    for layer, ranks in configs:
        core, factors, _ = decompose_layer(ctx.model, layer, tuple(ranks), n_iter_max or ctx.n_iter_max, ctx.tol)
        block, named = patch_for(layer, core, factors, prefix=f"L{layer}_")
        blocks.append(block)
        tensors.update(named)
    write_patch(SUBS / name / task, blocks, tensors, notes)


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


def previous_reference(ctx: Ctx) -> None:
    """The earlier gym's reference setting: layer 6, ranks (384,48,4), 5 HOOI sweeps, for every task."""
    for task in ctx.names:
        write_tucker("previous_reference", task, [(6, (384, 48, 4))], ctx, n_iter_max=5,
                     notes={"variant": "previous gym reference configuration"})


def weaker_low_rank(ctx: Ctx) -> None:
    """Reference layer, but the most aggressive ranks of the grid (64,16,2) -> CR 31.6."""
    for task in ctx.names:
        layer = ctx.selection[task]["layer"]
        write_tucker("weaker_low_rank", task, [(layer, (64, 16, 2))], ctx, notes={"variant": "low ranks"})


def shortcut_collapse_compress(ctx: Ctx) -> None:
    """Rank-(1,1,1) Tucker of the last layer: ~2782x compression, reasoning destroyed."""
    for task in ctx.names:
        write_tucker("shortcut_collapse_compress", task, [(11, (1, 1, 1))], ctx, notes={"variant": "collapse"})


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
        print(f"laser_mha layer {layer} done", flush=True)
    for name in ctx.names:
        best = max((e for e in log.values() if e[name]["finite"]),
                   key=lambda e: (e[name]["accuracy"], -e[name]["loss"]))
        layer, matrix, rank = best["layer"], best["matrix"], best["rank"]
        canonical = ctx.model.canonical(layer).double()
        u, s, vt = torch.linalg.svd(canonical[MATRICES.index(matrix)], full_matrices=False)
        tensors = {"US": u[:, :rank] * s[:rank], "Vt": vt[:rank]}
        block = {"layer": layer, "targets": [matrix], "equation": "ar,rb->ab", "operands": ["US", "Vt"]}
        write_patch(SUBS / "weaker_laser_mha" / name, [block], tensors,
                    {"variant": "LASER on one MHA matrix", "layer": layer, "matrix": matrix, "rank": rank,
                     "validation": best[name]})
        print(f"weaker_laser_mha {name}: layer {layer} {matrix} k={rank} val={best[name]}", flush=True)


def better_greedy_layers(ctx: Ctx, top_k: int = 4, max_layers: int = 3) -> None:
    """Greedy multi-layer TensorLLM: add further layers while validation accuracy strictly improves."""
    state_path = OUT / "greedy_layers.json"
    state = load_json(state_path, {})
    for name, task in ctx.tasks.items():
        if name in state and state[name].get("done"):
            continue
        sel = ctx.selection[name]
        configs = [(sel["layer"], tuple(sel["ranks"]))]
        current = sel["validation"]
        per_layer: dict[int, list] = {}
        for key, entry in ctx.log["conditions"].items():
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
                    overrides = merge(base, overrides_for(layer, ctx.recon(layer, ranks)))
                    res = task.evaluate(ctx.model, overrides)
                    if not res["finite"]:
                        continue
                    if best is None or (res["accuracy"], -res["loss"]) > (best[1]["accuracy"], -best[1]["loss"]):
                        best = ((layer, ranks), {"accuracy": res["accuracy"], "loss": res["loss"]})
            if best is None or best[1]["accuracy"] <= current["accuracy"]:
                break
            configs.append(best[0])
            current = best[1]
            history.append({"configs": [[c[0], list(c[1])] for c in configs], "validation": current})
            print(f"greedy {name}: {configs} val={current}", flush=True)
        write_tucker("better_greedy_layers", name, configs, ctx,
                     notes={"variant": "greedy multi-layer TensorLLM", "history": history})
        state[name] = {"done": True, "configs": [[c[0], list(c[1])] for c in configs], "validation": current}
        save_json(state_path, state)


def shortcut_supervised_ft(ctx: Ctx, epochs: int = 8, lr: float = 1e-4) -> None:
    """Out-of-scope shortcut: fine-tune the reference's Tucker core on visible labels (80/20 early stopping)."""
    torch.manual_seed(0)
    report = {}
    for name in ctx.names:
        sel = ctx.selection[name]
        layer, ranks = sel["layer"], tuple(sel["ranks"])
        core0, factors, _ = decompose_layer(ctx.model, layer, ranks, ctx.n_iter_max, ctx.tol)
        factors = [f.float() for f in factors]
        task = Task(name, ctx.device, token_budget=4096)
        prefix = task.prefix(ctx.model, layer)
        train_ids = [b for b in range(len(task.batches)) if b % 5 != 4]
        hold_ids = [b for b in range(len(task.batches)) if b % 5 == 4]

        def forward(core, b):
            batch = task.batches[b]
            canonical = torch.einsum(TUCKER_EQUATION, core, *factors).reshape(4, D_MODEL, D_MODEL)
            override = {m: Roberta.to_module(m, canonical[i]) for i, m in enumerate(MATRICES)}
            hidden = prefix[b].clone()
            for index in range(layer, N_LAYERS):
                hidden = ctx.model.layer(hidden, batch["keep"], index, override if index == layer else None)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return ctx.model.mask_log_probs(hidden[rows, batch["mask_pos"]]).index_select(1, task.candidates)

        def holdout_acc(core):
            correct = total = 0
            with torch.no_grad():
                for b in hold_ids:
                    pred = forward(core, b).argmax(1)
                    correct += int((pred == task.batches[b]["labels"]).sum())
                    total += len(task.batches[b]["index"])
            return 100.0 * correct / total

        core = core0.float().clone().requires_grad_(True)
        opt = torch.optim.Adam([core], lr=lr)
        best_acc, best_core = holdout_acc(core), core.detach().clone()
        history = [best_acc]
        rng = random.Random(0)
        for _ in range(epochs):
            order = list(train_ids)
            rng.shuffle(order)
            for b in order:
                loss = F.cross_entropy(forward(core, b), task.batches[b]["labels"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            acc = holdout_acc(core)
            history.append(acc)
            if acc > best_acc:
                best_acc, best_core = acc, core.detach().clone()
        block, tensors = patch_for(layer, best_core, factors)
        write_patch(SUBS / "shortcut_supervised_ft" / name, [block], tensors,
                    {"variant": "supervised fine-tuning of the core on visible labels", "holdout_history": history})
        report[name] = {"layer": layer, "ranks": list(ranks), "holdout_history": history}
        print(f"supervised_ft {name}: holdout {history}", flush=True)
    save_json(OUT / "supervised_ft.json", report)


def artifact_shortcuts(ctx: Ctx) -> None:
    """Patches that try to game the scorer's bookkeeping rather than the model."""
    ref = REFERENCE_DIR
    for name in ctx.names:
        # 1. forged metadata on the reference's own factors
        dst = SUBS / "shortcut_forged_metadata" / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(ref / name / "tensors.safetensors", dst / "tensors.safetensors")
        spec = json.loads((ref / name / "patch.json").read_text())
        spec.update({"compression_ratio": 1e6, "accuracy": 100.0, "loss": 0.0, "stored_params": 1})
        spec["notes"] = {"accuracy": 100.0, "compression_ratio": 1e6, "verdict": "pass"}
        (dst / "patch.json").write_text(json.dumps(spec, indent=2))

        # 2. NaN inside a factor
        from safetensors.torch import load_file

        tensors = load_file(str(ref / name / "tensors.safetensors"))
        core_name = [k for k in tensors if k.endswith("core")][0]
        tensors[core_name] = tensors[core_name].clone()
        tensors[core_name].view(-1)[0] = float("nan")
        write_patch(SUBS / "shortcut_nan_factor" / name, spec_blocks(ref / name), tensors)

        # 3. dense replacement of the reference layer (no compression)
        layer = ctx.selection[name]["layer"]
        dense = ctx.model.canonical(layer).float()
        write_patch(SUBS / "shortcut_dense_replacement" / name,
                    [{"layer": layer, "targets": list(MATRICES), "equation": "tmn->tmn", "operands": ["W"]}],
                    {"W": dense})

        # 4. one dense matrix broadcast to Q, K, V and O ("4x compression" by sharing)
        write_patch(SUBS / "shortcut_shared_dense" / name,
                    [{"layer": layer, "targets": list(MATRICES), "equation": "t,mn->tmn", "operands": ["ones", "W"]}],
                    {"ones": torch.ones(4), "W": dense[2]})

        # 5. contraction that does not produce the declared matrices
        write_patch(SUBS / "shortcut_bad_contraction" / name,
                    [{"layer": layer, "targets": list(MATRICES), "equation": "ab,bc->ac", "operands": ["A", "B"]}],
                    {"A": torch.zeros(768, 8), "B": torch.zeros(8, 768)})

    # 6. only one task submitted
    dst = SUBS / "shortcut_missing_experiment" / "fever_roberta_binary"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ref / "fever_roberta_binary", dst, dirs_exist_ok=True)
    # 7. empty submission directory
    (SUBS / "shortcut_empty").mkdir(parents=True, exist_ok=True)
    # 8. a directory that claims the evaluator-owned baseline key (holds a collapse patch)
    for name in ctx.names:
        write_tucker("roberta_original", name, [(11, (1, 1, 1))], ctx)


def spec_blocks(patch_dir: Path) -> list:
    return json.loads((patch_dir / "patch.json").read_text())["blocks"]


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
    "shortcut_collapse_compress": shortcut_collapse_compress,
    "shortcut_supervised_ft": shortcut_supervised_ft,
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
