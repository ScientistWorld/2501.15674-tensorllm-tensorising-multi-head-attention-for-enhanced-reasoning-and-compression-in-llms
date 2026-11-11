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
    detensorize,
    mode_dot,
    overrides_for,
    patch_for,
    reconstruct_canonical,
    tensorize,
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
        _, factors, _ = decompose_layer(ctx.model, layer, tuple(ranks), n_iter_max or ctx.n_iter_max, ctx.tol)
        block, named = patch_for(layer, factors, prefix=f"L{layer}_")
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
        # rank-k truncated SVD == projection of the matrix onto its top-k left singular vectors
        tensors = {"U": u[:, :rank]}
        block = {"layer": layer, "targets": [matrix], "layout": "stack", "factors": {"0": "U"}}
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
            configs = [(layer, tuple(ranks)) for layer, ranks in state[name]["configs"]]
            write_tucker("better_greedy_layers", name, configs, ctx,
                         notes={"variant": "greedy multi-layer TensorLLM", "validation": state[name]["validation"]})
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


def shortcut_supervised_subspace(ctx: Ctx, epochs: int = 8, lr: float = 3e-4) -> None:
    """Strongest supervised attack the projection format still allows: gradient-train the reference's
    three factor matrices (the kept subspaces) on visible labels, 80/20 split with early stopping.
    The rebuilt weights stay an orthogonal projection of the original weights."""
    torch.manual_seed(0)
    report = {}
    for name in ctx.names:
        sel = ctx.selection[name]
        layer, ranks = sel["layer"], tuple(sel["ranks"])
        _, factors0, _ = decompose_layer(ctx.model, layer, ranks, ctx.n_iter_max, ctx.tol)
        params = [f.float().clone().requires_grad_(True) for f in factors0]
        w_all = tensorize(ctx.model.canonical(layer)).float()
        task = Task(name, ctx.device, token_budget=4096)
        prefix = task.prefix(ctx.model, layer)
        train_ids = [b for b in range(len(task.batches)) if b % 5 != 4]
        hold_ids = [b for b in range(len(task.batches)) if b % 5 == 4]

        def rebuild(factors):
            bases = [torch.linalg.qr(f)[0] for f in factors]
            core = w_all
            for mode, q in enumerate(bases):
                core = mode_dot(core, q.T, mode)
            rec = core
            for mode, q in enumerate(bases):
                rec = mode_dot(rec, q, mode)
            canonical = detensorize(rec)
            return {m: Roberta.to_module(m, canonical[i]) for i, m in enumerate(MATRICES)}

        def forward(override, b):
            batch = task.batches[b]
            hidden = prefix[b].clone()
            for index in range(layer, N_LAYERS):
                hidden = ctx.model.layer(hidden, batch["keep"], index, override if index == layer else None)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return ctx.model.mask_log_probs(hidden[rows, batch["mask_pos"]]).index_select(1, task.candidates)

        def holdout_acc(factors):
            correct = total = 0
            with torch.no_grad():
                override = rebuild(factors)
                for b in hold_ids:
                    pred = forward(override, b).argmax(1)
                    correct += int((pred == task.batches[b]["labels"]).sum())
                    total += len(task.batches[b]["index"])
            return 100.0 * correct / total

        opt = torch.optim.Adam(params, lr=lr)
        best_acc, best = holdout_acc(params), [f.detach().clone() for f in params]
        history = [best_acc]
        rng = random.Random(0)
        for _ in range(epochs):
            order = list(train_ids)
            rng.shuffle(order)
            for b in order:
                loss = F.cross_entropy(forward(rebuild(params), b), task.batches[b]["labels"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            acc = holdout_acc(params)
            history.append(acc)
            if acc > best_acc:
                best_acc, best = acc, [f.detach().clone() for f in params]
        block, tensors = patch_for(layer, best)
        write_patch(SUBS / "shortcut_supervised_subspace" / name, [block], tensors,
                    {"variant": "factor subspaces trained on visible labels", "holdout_history": history})
        report[name] = {"layer": layer, "ranks": list(ranks), "holdout_history": history}
        print(f"supervised_subspace {name}: holdout {history}", flush=True)
    save_json(OUT / "supervised_subspace.json", report)


def artifact_shortcuts(ctx: Ctx) -> None:
    """Patches that try to game the scorer's bookkeeping rather than the model."""
    from safetensors.torch import load_file

    ref = REFERENCE_DIR
    for name in ctx.names:
        layer = ctx.selection[name]["layer"]
        ref_blocks = spec_blocks(ref / name)
        # 1. forged metadata on the reference's own factors
        dst = SUBS / "shortcut_forged_metadata" / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(ref / name / "tensors.safetensors", dst / "tensors.safetensors")
        spec = json.loads((ref / name / "patch.json").read_text())
        spec.update({"compression_ratio": 1e6, "accuracy": 100.0, "loss": 0.0, "stored_params": 1})
        spec["notes"] = {"accuracy": 100.0, "compression_ratio": 1e6, "verdict": "pass"}
        spec["blocks"][0]["core"] = "trust me"
        (dst / "patch.json").write_text(json.dumps(spec, indent=2))

        # 2. NaN inside a factor
        tensors = load_file(str(ref / name / "tensors.safetensors"))
        tensors["U_model"] = tensors["U_model"].clone()
        tensors["U_model"][0, 0] = float("nan")
        write_patch(SUBS / "shortcut_nan_factor" / name, ref_blocks, tensors)

        # 3. "dense replacement": full-rank identity factors on every mode (no compression)
        write_patch(SUBS / "shortcut_dense_replacement" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads",
                      "factors": {"0": "I_model", "1": "I_head", "2": "I_type"}}],
                    {"I_model": torch.eye(768), "I_head": torch.eye(64), "I_type": torch.eye(4)})

        # 4. factor that claims rank 64 but spans one direction
        column = torch.randn(768, 1, generator=torch.Generator().manual_seed(0))
        write_patch(SUBS / "shortcut_rank_deficient" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads", "factors": {"0": "U"}}],
                    {"U": column.repeat(1, 64)})

        # 5. factor whose shape does not match the declared layout
        write_patch(SUBS / "shortcut_bad_shape" / name,
                    [{"layer": layer, "targets": list(MATRICES), "layout": "heads", "factors": {"0": "U"}}],
                    {"U": torch.randn(700, 8, generator=torch.Generator().manual_seed(0))})

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
    "shortcut_supervised_subspace": shortcut_supervised_subspace,
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
