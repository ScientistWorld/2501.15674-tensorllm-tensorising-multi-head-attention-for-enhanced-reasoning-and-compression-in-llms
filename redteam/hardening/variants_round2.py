#!/usr/bin/env python3
"""Hardening round 2: new variants and shortcuts, scored through the unchanged scorers.

Written next to the round-1 submissions under ``outputs/hardening/submissions/``
in the submission format a future method uses.  Selection inside every variant
uses only the visible split; nothing here reads held-out data.

* ``compress64_L<l>`` (12 methods): heavy compression of one layer.  The
  shared-factor Tucker of layer ``l`` at ranks giving >= 64x compression,
  choosing the ranks with the best visible accuracy.  These probe whether a
  method can buy rank with compression instead of reasoning (round-1 gate:
  credit whenever accuracy is above the unmodified model by any margin).
* ``shortcut_max_compression``: the best of those by visible accuracy.
* ``variant_head_groups``: the reference layer and ranks, but the 12 heads are
  split into two groups of 6 by a data-free spectral clustering of their
  QK and VO circuits, each group with its own shared factors.  Exercises the
  new ``heads`` field; pre-declared, no selection.
* ``variant_qk_vo_split``: the reference layer, query/key and value/output in
  two separate multi-head tensors (shared factors within each circuit).
  Exercises target grouping; pre-declared, no selection.
* ``check_explicit_layout``: the reference written with an explicit axis-group
  layout and an explicit list of all heads -- must score exactly as the reference.
* ``shortcut_overlapping_heads``, ``shortcut_bad_axes``, ``shortcut_slice_bloat``:
  malformed uses of the new fields; all must fail.

    python redteam/hardening/variants_round2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.mlm_tasks import (  # noqa: E402
    HEAD_DIM,
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
from method.tensorllm_roberta import (  # noqa: E402
    compression_ratio,
    decompose_layer,
    overrides_for,
    patch_for,
    reconstruct_canonical,
    valid_ranks,
)

OUT = ROOT / "outputs" / "hardening"
SUBS = OUT / "submissions"
REFERENCE_DIR = ROOT / "outputs" / "submissions" / "tensorllm_roberta"
COMPRESS_LOG = OUT / "compress64_log.json"
HIGH_CR_GRID = [
    (16, 8, 2), (16, 16, 2), (16, 24, 2), (16, 16, 4), (24, 16, 2),
    (24, 24, 2), (32, 8, 2), (32, 8, 3), (32, 16, 1), (32, 24, 1),
]


def compress64(model, tasks) -> None:
    grid = [r for r in HIGH_CR_GRID if valid_ranks(r) and compression_ratio(r) >= 64.0]
    log = load_json(COMPRESS_LOG, {"conditions": {}})
    conditions = log["conditions"]
    started = time.time()
    for layer in range(N_LAYERS):
        for ranks in grid:
            key = f"L{layer}_r{ranks[0]}-{ranks[1]}-{ranks[2]}"
            entry = conditions.get(key, {})
            if all(name in entry for name in tasks):
                continue
            core, factors, info = decompose_layer(model, layer, ranks, 100, 1e-8)
            overrides = overrides_for(layer, reconstruct_canonical(core, factors))
            entry.update({"layer": layer, "ranks": list(ranks), "compression_ratio": compression_ratio(ranks), **info})
            for name, task in tasks.items():
                res = task.evaluate(model, overrides)
                entry[name] = {"accuracy": res["accuracy"], "loss": res["loss"], "finite": res["finite"]}
            conditions[key] = entry
            save_json(COMPRESS_LOG, log)
            print(f"compress64 {key} CR={entry['compression_ratio']:.1f} "
                  + " ".join(f"{n[:5]}={entry[n]['accuracy']:.2f}" for n in tasks)
                  + f" ({time.time() - started:.0f}s)", flush=True)
    for name in tasks:
        best_overall = None
        for layer in range(N_LAYERS):
            cands = [e for e in conditions.values() if e["layer"] == layer and e[name]["finite"]]
            if not cands:
                continue
            best = max(cands, key=lambda e: (e[name]["accuracy"], -e[name]["loss"]))
            write_patch(SUBS / f"compress64_L{layer}" / name, [patch_for(layer, best["ranks"])],
                        {"variant": "heavy compression of one layer (>= 64x), best visible accuracy",
                         "validation": best[name], "compression_ratio": best["compression_ratio"]})
            if best_overall is None or (best[name]["accuracy"], -best[name]["loss"]) > (
                    best_overall[name]["accuracy"], -best_overall[name]["loss"]):
                best_overall = best
        write_patch(SUBS / "shortcut_max_compression" / name,
                    [patch_for(best_overall["layer"], best_overall["ranks"])],
                    {"variant": ">= 64x compression with the best visible accuracy over all layers",
                     "validation": best_overall[name], "layer": best_overall["layer"],
                     "ranks": best_overall["ranks"]})
        print(f"shortcut_max_compression {name}: L{best_overall['layer']} {best_overall['ranks']} "
              f"val={best_overall[name]}", flush=True)


def head_groups(model: Roberta, layer: int) -> list[list[int]]:
    """Two groups of 6 heads from a spectral split of QK / VO circuit similarity (weights only)."""
    canon = model.canonical(layer).double()  # [4, d, h*dv]
    descriptors = []
    for head in range(N_HEADS):
        cols = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        qk = canon[0][:, cols] @ canon[1][:, cols].T  # d x d bilinear form of the head
        vo = canon[2][:, cols] @ canon[3][:, cols].T  # d x d value-output map of the head
        descriptors.append(torch.cat([(qk / qk.norm()).flatten(), (vo / vo.norm()).flatten()]))
    desc = torch.stack(descriptors)
    desc = desc / desc.norm(dim=1, keepdim=True)
    affinity = (desc @ desc.T).clamp(min=0.0)
    affinity.fill_diagonal_(0.0)
    laplacian = torch.diag(affinity.sum(1)) - affinity
    _, vectors = torch.linalg.eigh(laplacian)
    fiedler = vectors[:, 1].tolist()
    order = sorted(range(N_HEADS), key=lambda h: (fiedler[h], h))
    first = sorted(order[: N_HEADS // 2])
    second = sorted(order[N_HEADS // 2:])
    return sorted([first, second])


def structural_variants(model, selection: dict, names) -> None:
    for name in names:
        layer = selection[name]["layer"]
        r1, r2, r3 = selection[name]["ranks"]
        groups = head_groups(model, layer)
        write_patch(SUBS / "variant_head_groups" / name, [
            {"layer": layer, "targets": list(MATRICES), "heads": group, "layout": "heads",
             "ranks": {"0": r1, "1": r2, "2": r3}} for group in groups
        ], {"variant": "reference ranks, heads split in two data-free clusters", "groups": groups})
        print(f"variant_head_groups {name}: layer {layer} groups {groups}", flush=True)
        write_patch(SUBS / "variant_qk_vo_split" / name, [
            {"layer": layer, "targets": ["query", "key"], "layout": "heads", "ranks": {"0": r1, "1": r2}},
            {"layer": layer, "targets": ["value", "output"], "layout": "heads", "ranks": {"0": r1, "1": r2}},
        ], {"variant": "reference ranks, query/key and value/output tensorised separately"})
        write_patch(SUBS / "check_explicit_layout" / name, [
            {"layer": layer, "targets": list(MATRICES), "heads": list(range(N_HEADS)),
             "layout": [["m"], ["d"], ["t"], ["h"]], "ranks": {"0": r1, "1": r2, "2": r3}}
        ], {"variant": "the reference, spelled with an explicit layout and head list"})
        write_patch(SUBS / "shortcut_overlapping_heads" / name, [
            {"layer": layer, "targets": list(MATRICES), "layout": "heads", "ranks": {"0": r1, "1": r2, "2": r3}},
            {"layer": layer, "targets": ["query"], "heads": [0], "layout": "stack", "ranks": {"0": 4}},
        ])
        write_patch(SUBS / "shortcut_bad_axes" / name, [
            {"layer": layer, "targets": list(MATRICES), "layout": [["m"], ["d"], ["t"]], "ranks": {"0": r1}}
        ])
        # every head of every matrix as its own block: individually small, together larger than dense
        write_patch(SUBS / "shortcut_slice_bloat" / name, [
            {"layer": layer, "targets": list(MATRICES), "heads": [head], "layout": "heads", "ranks": {"0": 256}}
            for head in range(N_HEADS)
        ])


def main() -> None:
    SUBS.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = Roberta(device)
    names = task_names()
    tasks = {name: Task(name, device) for name in names}
    selection = json.loads((REFERENCE_DIR / "selection.json").read_text())
    started = time.time()
    compress64(model, tasks)
    structural_variants(model, selection, names)
    print(f"round-2 variants written in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
