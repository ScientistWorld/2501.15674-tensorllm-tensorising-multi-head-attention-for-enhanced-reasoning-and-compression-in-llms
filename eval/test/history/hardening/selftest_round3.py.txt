#!/usr/bin/env python3
"""Round-3 self-test (runs inside the job before the search; exit 1 stops the job).

1. The WikidataQA data files: visible (data/tasks) and held-out (eval/test/data)
   splits are disjoint, have the sizes of LASER's protocol and their hashes match
   the manifests.
2. Both scorers tokenize every WikidataQA row (five masks; LASER's answer-length
   rule finds every answer), and the method-side task (baseline/mlm_tasks.py)
   computes exactly the scorer's accuracy and loss on the visible split, for the
   unmodified model and for one shared-factor Tucker structure.
3. Prints LASER-protocol diagnostics: example generations, how often LASER's
   answer-length rule differs from tokenizing " <answer>" on its own, and the
   unmodified model's visible accuracy / loss (the paper reports 28.0 on its
   test portion).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK = "wikidata_qa_roberta"
FAILURES: list[str] = []


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def main() -> int:
    test_mod = load("scorer_test_r3", ROOT / "eval" / "test" / "evaluate.py")
    train_mod = load("scorer_train_r3", ROOT / "eval" / "train" / "evaluate.py")
    from baseline.mlm_tasks import Roberta, Task, pick_device
    from method.tensorllm_roberta import decompose_layer, overrides_for, patch_for, reconstruct_canonical

    # 1. data files
    visible_meta = json.loads((ROOT / "data" / "tasks" / "tasks.json").read_text())["tasks"]
    test_meta = json.loads((ROOT / "eval" / "test" / "data" / "MANIFEST.json").read_text())["tasks"]
    check(list(visible_meta) == ["bios_profession_roberta", "fever_roberta_binary", TASK],
          f"visible tasks {list(visible_meta)}")
    check(list(test_meta) == ["bios_profession_roberta", "fever_roberta_binary", TASK], f"held-out tasks {list(test_meta)}")
    vis_rows = [json.loads(l) for l in (ROOT / "data" / "tasks" / visible_meta[TASK]["file"]).open(encoding="utf-8")]
    test_rows = [json.loads(l) for l in (ROOT / "eval" / "test" / "data" / test_meta[TASK]["file"]).open(encoding="utf-8")]
    n_full = visible_meta[TASK]["n_full_dataset"]
    check(len(vis_rows) == int(0.2 * n_full) and len(vis_rows) + len(test_rows) + len(
        test_meta[TASK]["dropped_duplicates_of_visible_prompts"]) == n_full,
          f"split sizes: visible {len(vis_rows)} + held-out {len(test_rows)} of {n_full}")
    check(not ({r["id"] for r in vis_rows} & {r["id"] for r in test_rows})
          and not ({r["question"] for r in vis_rows} & {r["question"] for r in test_rows}),
          "visible and held-out WikidataQA share no id and no question")
    for meta, folder in ((visible_meta, ROOT / "data" / "tasks"), (test_meta, ROOT / "eval" / "test" / "data")):
        for name, entry in meta.items():
            digest = hashlib.sha256((folder / entry["file"]).read_bytes()).hexdigest()
            check(digest == entry["sha256"], f"{folder.relative_to(ROOT)}/{entry['file']} hash")

    device = test_mod.configure_torch()
    model = test_mod.FixedRoberta(device)
    budget = 32768 if device.type == "cuda" else 4096

    # 2. both scorers prepare every row; method-side task == scorer
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(test_mod.MODEL_DIR / "tokenizer.json"))
    for module, rows, label in ((test_mod, test_rows, "held-out"), (train_mod, vis_rows, "visible")):
        try:
            prepared = module.tokenize_generation(TASK, rows, tokenizer, device, budget)
            n_loss = sum(int(b["count"].sum()) for b in prepared["loss_batches"])
            check(True, f"{module.__name__}: {label} split tokenized ({len(rows)} rows, {n_loss} answer tokens)")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{module.__name__}: {label} split tokenized ({exc!r})")
    differs = 0
    for row in vis_rows + test_rows:
        claim = tokenizer.encode(row["question"] + " " + row["answer"], add_special_tokens=False).ids
        k = test_mod.answer_token_count(tokenizer, claim, row["answer"])
        alone = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False).ids
        differs += int(claim[len(claim) - k:] != alone)
    print(f"info LASER answer-length rule differs from tokenizing ' <answer>' alone on {differs} rows", flush=True)

    prepared = train_mod.tokenize_generation(TASK, vis_rows, tokenizer, device, budget)
    method_model = Roberta(pick_device())
    task = Task(TASK, method_model.device)
    with tempfile.TemporaryDirectory() as tmp_name:
        patch_dir = Path(tmp_name) / "p"
        patch_dir.mkdir()
        layer, ranks = 6, (256, 48, 4)
        (patch_dir / "patch.json").write_text(json.dumps({"format": train_mod.PATCH_FORMAT,
                                                          "blocks": [patch_for(layer, ranks)]}))
        overrides, info = train_mod.materialize_patch(patch_dir, model)
        by_layer = {}
        for (lay, matrix), value in overrides.items():
            by_layer.setdefault(lay, {})[matrix] = train_mod.canonical_to_module(matrix, value).to(device)
        records = train_mod.score_experiment(model, prepared, {"base": {}, "tucker": by_layer})
        core, factors, _ = decompose_layer(method_model, layer, ranks, 100, 1e-8)
        method_over = overrides_for(layer, reconstruct_canonical(core, factors))
        for key, over in (("base", None), ("tucker", method_over)):
            scorer = train_mod.summarize(records[key])
            mine = task.evaluate(method_model, over)
            tol_acc = 1e-9 if key == "base" else 0.15  # the method rebuilds with einsum, the scorer with mode products
            check(abs(scorer["accuracy"] - mine["accuracy"]) <= tol_acc and abs(scorer["loss"] - mine["loss"]) <= (
                1e-6 if key == "base" else 0.01),
                  f"method-side task == scorer on the visible split ({key}): "
                  f"acc {mine['accuracy']:.3f} vs {scorer['accuracy']:.3f}, loss {mine['loss']:.4f} vs {scorer['loss']:.4f}")
        base = records["base"]
        laser = train_mod.summarize(base, "nll_laser")["loss"]
        print(f"info unmodified roberta-base, visible WikidataQA: accuracy {train_mod.summarize(base)['accuracy']:.2f}, "
              f"loss {train_mod.summarize(base)['loss']:.3f} (LASER convention {laser:.3f}); "
              f"structure L{layer} {ranks}: accuracy {train_mod.summarize(records['tucker'])['accuracy']:.2f} "
              f"(CR {info['compression_ratio']:.2f})", flush=True)
        for i in range(0, len(vis_rows), max(1, len(vis_rows) // 12)):
            print(f"info   {vis_rows[i]['question']!r} -> gold {vis_rows[i]['answer']!r}; "
                  f"generated {base['pred'][i]!r} correct={base['correct'][i]}", flush=True)

    print(f"\nself-test round 3: {len(FAILURES)} failure(s)", flush=True)
    for failure in FAILURES:
        print("  FAIL", failure, flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    torch.manual_seed(0)
    sys.exit(main())
