#!/usr/bin/env python3
"""Hardening round 3: the BIG-bench WikidataQA experiment.

Everything is written under ``outputs/hardening/submissions/<name>/<task>/`` in
the submission format a future method uses and scored by the unchanged
scorers.  Selection inside every variant uses only the visible split.

Round-1 and round-2 variants are extended to the new task (their FEVER and
Bias in Bios patches are already on disk and unchanged):
  previous_reference, weaker_low_rank, weaker_laser_mha, better_greedy_layers,
  greedy_layers_nomargin, shortcut_collapse_compress, the artifact shortcuts,
  compress64_L<l> / shortcut_max_compression, variant_head_groups,
  variant_qk_vo_split and the malformed-field shortcuts.

New in round 3, for every task:
  * ``greedy_layers_m05``: multi-layer TensorLLM, greedily adding a layer while
    visible accuracy rises by >= 0.5 points (up to 4 layers) -- a second
    candidate for a better method;
  * ``baseline_laser_mha_all4``: the paper's LASER comparison (Table III,
    case 2) restricted to attention: a truncated SVD at one rank of each of the
    four attention matrices of one layer, layer and rank chosen on the visible
    split;
  * ``ablation_single_<matrix>``: the paper's Table IV ablation -- the
    multi-head tensorisation and shared-factor Tucker applied to ONE of the
    query / key / value / output matrices; layer and ranks chosen on the
    visible split.

    python redteam/hardening/variants_round3.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "redteam" / "hardening"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import variants as r1  # noqa: E402
import variants_round2 as r2  # noqa: E402
from baseline.mlm_tasks import (  # noqa: E402
    MATRICES,
    N_HEADS,
    N_LAYERS,
    Roberta,
    Task,
    load_json,
    pick_device,
    save_json,
    task_names,
    write_patch,
)

NEW_TASK = "wikidata_qa_roberta"
OUT = ROOT / "outputs" / "hardening"
SUBS = OUT / "submissions"
DONE = OUT / "variants_round3_done.json"
LASER_RANKS = r1.LASER_RANKS
SINGLE_GRID = [(64, 32), (128, 32), (256, 32), (64, 48), (128, 48), (256, 48)]


class Ctx(r1.Ctx):
    """Round-1 context restricted to the given tasks (cached results of the others are reused)."""

    def __init__(self, names, n_iter_max: int = 100, tol: float = 1e-8):  # noqa: D107
        self.device = pick_device()
        self.model = Roberta(self.device)
        self.names = list(names)
        self.tasks = {name: Task(name, self.device) for name in self.names}
        self.selection = json.loads((r1.REFERENCE_DIR / "selection.json").read_text())
        self.log = load_json(r1.SEARCH_LOG, {"conditions": {}})
        self.n_iter_max = n_iter_max
        self.tol = tol
        self._recon = {}


def best_of(entries, name):
    finite = [e for e in entries if e.get(name, {}).get("finite")]
    return max(finite, key=lambda e: (e[name]["accuracy"], -e[name]["loss"]))


def laser_single(ctx: Ctx) -> None:
    """weaker_laser_mha for tasks without cached results (only missing tasks are evaluated)."""
    log_path = OUT / "laser_mha_log.json"
    log = load_json(log_path, {})
    for layer in range(N_LAYERS):
        canonical = ctx.model.canonical(layer).double()
        for m_index, matrix in enumerate(MATRICES):
            u, s, vt = torch.linalg.svd(canonical[m_index], full_matrices=False)
            for rank in LASER_RANKS:
                key = f"L{layer}_{matrix}_k{rank}"
                entry = log.setdefault(key, {"layer": layer, "matrix": matrix, "rank": rank})
                missing = [name for name in ctx.names if name not in entry]
                if not missing:
                    continue
                approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
                override = {layer: {matrix: Roberta.to_module(matrix, approx.float()).contiguous()}}
                for name in missing:
                    res = ctx.tasks[name].evaluate(ctx.model, override)
                    entry[name] = {"accuracy": res["accuracy"], "loss": res["loss"], "finite": res["finite"]}
            save_json(log_path, log)
    for name in ctx.names:
        best = best_of(log.values(), name)
        block = {"layer": best["layer"], "targets": [best["matrix"]], "layout": "stack", "ranks": {"0": best["rank"]}}
        write_patch(SUBS / "weaker_laser_mha" / name, [block],
                    {"variant": "LASER on one MHA matrix", "validation": best[name]})
        print(f"weaker_laser_mha {name}: L{best['layer']} {best['matrix']} k={best['rank']} val={best[name]}",
              flush=True)


def laser_all4(ctx: Ctx) -> None:
    """LASER on all four attention matrices of one layer (separate truncated SVDs, one rank)."""
    log_path = OUT / "laser_all4_log.json"
    log = load_json(log_path, {})
    for layer in range(N_LAYERS):
        canonical = ctx.model.canonical(layer).double()
        svds = [torch.linalg.svd(canonical[i], full_matrices=False) for i in range(len(MATRICES))]
        for rank in LASER_RANKS:
            key = f"L{layer}_k{rank}"
            entry = log.setdefault(key, {"layer": layer, "rank": rank})
            missing = [name for name in ctx.names if name not in entry]
            if not missing:
                continue
            override = {layer: {}}
            for (u, s, vt), matrix in zip(svds, MATRICES):
                approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
                override[layer][matrix] = Roberta.to_module(matrix, approx.float()).contiguous()
            for name in missing:
                res = ctx.tasks[name].evaluate(ctx.model, override)
                entry[name] = {"accuracy": res["accuracy"], "loss": res["loss"], "finite": res["finite"]}
            save_json(log_path, log)
    for name in ctx.names:
        best = best_of(log.values(), name)
        blocks = [{"layer": best["layer"], "targets": [matrix], "layout": "stack", "ranks": {"0": best["rank"]}}
                  for matrix in MATRICES]
        write_patch(SUBS / "baseline_laser_mha_all4" / name, blocks,
                    {"variant": "LASER (truncated SVD) of all four attention matrices of one layer",
                     "validation": best[name], "layer": best["layer"], "rank": best["rank"]})
        print(f"baseline_laser_mha_all4 {name}: L{best['layer']} k={best['rank']} val={best[name]}", flush=True)


def single_matrix_tucker(w: torch.Tensor, r1_: int, r2_: int) -> torch.Tensor:
    """Shared-factor Tucker of ONE matrix's heads: [d_model, d_v, h], factors on modes 0, 1 (head mode kept)."""
    from method.tensorllm_roberta import leading_left_singular_vectors, mode_dot, unfold
    import math

    tensor = w.reshape(768, N_HEADS, 64).permute(0, 2, 1).contiguous().double()  # [m, d, h]
    ranks = {0: r1_, 1: r2_}
    norm_sq = float((tensor * tensor).sum())
    factors = {m: leading_left_singular_vectors(unfold(tensor, m), r) for m, r in ranks.items()}
    errors = []
    for iteration in range(100):
        for mode in ranks:
            projected = tensor
            for other in ranks:
                if other != mode:
                    projected = mode_dot(projected, factors[other].T, other)
            factors[mode] = leading_left_singular_vectors(unfold(projected, mode), ranks[mode])
        core = tensor
        for mode in ranks:
            core = mode_dot(core, factors[mode].T, mode)
        errors.append(math.sqrt(abs(norm_sq - float((core * core).sum()))) / math.sqrt(norm_sq))
        if iteration > 1 and abs(errors[-2] - errors[-1]) < 1e-8:
            break
    core = tensor
    for mode in ranks:
        core = mode_dot(core, factors[mode].T, mode)
    rebuilt = core
    for mode in ranks:
        rebuilt = mode_dot(rebuilt, factors[mode], mode)
    return rebuilt.permute(0, 2, 1).reshape(768, 768).float()


def ablation_single(ctx: Ctx) -> None:
    """Table IV: the method's tensorisation applied to one of the four attention matrices."""
    log_path = OUT / "ablation_single_log.json"
    log = load_json(log_path, {})
    started = time.time()
    for layer in range(N_LAYERS):
        canonical = ctx.model.canonical(layer)
        for m_index, matrix in enumerate(MATRICES):
            for r1_, r2_ in SINGLE_GRID:
                key = f"L{layer}_{matrix}_r{r1_}-{r2_}"
                entry = log.setdefault(key, {"layer": layer, "matrix": matrix, "ranks": [r1_, r2_]})
                missing = [name for name in ctx.names if name not in entry]
                if not missing:
                    continue
                rebuilt = single_matrix_tucker(canonical[m_index], r1_, r2_)
                override = {layer: {matrix: Roberta.to_module(matrix, rebuilt).contiguous()}}
                for name in missing:
                    res = ctx.tasks[name].evaluate(ctx.model, override)
                    entry[name] = {"accuracy": res["accuracy"], "loss": res["loss"], "finite": res["finite"]}
                save_json(log_path, log)
        print(f"ablation_single: layer {layer} done ({time.time() - started:.0f}s)", flush=True)
    for matrix in MATRICES:
        for name in ctx.names:
            best = best_of([e for e in log.values() if e["matrix"] == matrix], name)
            r1_, r2_ = best["ranks"]
            # heads layout of one matrix: [d_model, d_v, 1, h]; shared factors on modes 0 and 1
            block = {"layer": best["layer"], "targets": [matrix], "layout": "heads", "ranks": {"0": r1_, "1": r2_}}
            write_patch(SUBS / f"ablation_single_{matrix}" / name, [block],
                        {"variant": f"multi-head shared-factor Tucker of the {matrix} matrix only (Table IV)",
                         "validation": best[name]})
            print(f"ablation_single_{matrix} {name}: L{best['layer']} r={best['ranks']} val={best[name]}",
                  flush=True)


def greedy_m05(ctx: Ctx) -> None:
    """Multi-layer TensorLLM, adding layers while visible accuracy rises by >= 0.5 points (<= 4 layers)."""
    state_path = OUT / "greedy_layers_m05.json"
    state = load_json(state_path, {})
    for name, task in ctx.tasks.items():
        if not state.get(name, {}).get("done"):
            configs, current, history = r1._greedy(ctx, name, task, 0.5, 4, 4)
            state[name] = {"done": True, "configs": [[c[0], list(c[1])] for c in configs],
                           "validation": current, "history": history}
            save_json(state_path, state)
        configs = [(layer, tuple(ranks)) for layer, ranks in state[name]["configs"]]
        r1.write_tucker("greedy_layers_m05", name, configs,
                        notes={"variant": "greedy multi-layer TensorLLM, min gain 0.5, up to 4 layers",
                               "validation": state[name]["validation"]})


def main() -> None:
    SUBS.mkdir(parents=True, exist_ok=True)
    r1.copy_reference()
    done = load_json(DONE, {})
    all_names = task_names()
    started = time.time()

    new_ctx = Ctx([NEW_TASK])
    steps = [
        ("previous_reference", lambda: r1.previous_reference(new_ctx)),
        ("weaker_low_rank", lambda: r1.weaker_low_rank(new_ctx)),
        ("shortcut_collapse_compress", lambda: r1.shortcut_collapse_compress(new_ctx)),
        ("artifact_shortcuts", lambda: r1.artifact_shortcuts(new_ctx)),
        ("weaker_laser_mha", lambda: laser_single(new_ctx)),
        ("better_greedy_layers", lambda: r1.better_greedy_layers(new_ctx)),
        ("greedy_layers_nomargin", lambda: r1.greedy_layers_nomargin(new_ctx)),
        ("compress64", lambda: r2.compress64(new_ctx.model, new_ctx.tasks)),
        ("structural_variants", lambda: r2.structural_variants(new_ctx.model, new_ctx.selection, [NEW_TASK])),
    ]
    for key, fn in steps:
        if done.get(key):
            print(f"skip {key} (done)", flush=True)
            continue
        t0 = time.time()
        print(f"=== {key} [{NEW_TASK}]", flush=True)
        fn()
        done[key] = round(time.time() - t0, 1)
        save_json(DONE, done)
    del steps, new_ctx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_ctx = Ctx(all_names)
    for key, fn in [("baseline_laser_mha_all4", laser_all4), ("ablation_single", ablation_single),
                    ("greedy_layers_m05", greedy_m05)]:
        if done.get(key):
            print(f"skip {key} (done)", flush=True)
            continue
        t0 = time.time()
        print(f"=== {key} {all_names}", flush=True)
        fn(all_ctx)
        done[key] = round(time.time() - t0, 1)
        save_json(DONE, done)
    print(f"round-3 variants written in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
