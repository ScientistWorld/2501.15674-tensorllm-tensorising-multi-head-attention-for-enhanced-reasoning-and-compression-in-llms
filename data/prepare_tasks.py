#!/usr/bin/env python3
"""Build the public (visible) validation splits for the two reasoning tasks.

Both tasks follow the preprocessing of LASER (Sharma et al., 2024), which the
paper says it adopts:

* ``fever_roberta_binary`` -- FEVER v1.0 ``paper_dev`` followed by
  ``paper_test``; NOT ENOUGH INFO claims dropped; duplicate claims merged and
  claims that occur with conflicting labels removed (13,086 claims).
* ``bios_profession_roberta`` -- Bias in Bios ``dev`` split restricted to the
  eight professions whose names are single RoBERTa tokens; each biography is
  cut to its last 50 tokens (19,2xx biographies).

Only the first 20% of each ordered dataset (the validation portion used for
hyperparameter selection) is written here.  Held-out evaluation examples are
not produced by this script.

Usage (inside the container, after ``scripts/download.sh`` fetched the raw
files):
    python data/prepare_tasks.py --raw-dir <dir with raw files>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "tasks"
MODEL_DIR = ROOT / "data" / "downloads" / "roberta-base"
VAL_FRACTION_DENOM = 5  # first N // 5 examples (20%)

BIOS_OCCUPATIONS = ["journalist", "poet", "composer", "model", "teacher", "architect", "painter", "professor"]
BIOS_PROFESSION_IDS = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor", 4: "comedian", 5: "composer",
    6: "dentist", 7: "dietitian", 8: "dj", 9: "filmmaker", 10: "interior designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal", 16: "pastor", 17: "personal trainer",
    18: "photographer", 19: "physician", 20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software engineer", 25: "surgeon", 26: "teacher", 27: "yoga teacher",
}
BIOS_MAX_TOKENS = 50

TASKS = {
    "bios_profession_roberta": {
        "candidates": [f" {name}" for name in BIOS_OCCUPATIONS],
        "file": "bios_profession_val.jsonl",
        "description": "Bias in Bios profession prediction (8 single-token professions).",
    },
    "fever_roberta_binary": {
        "candidates": [" false", " true"],
        "file": "fever_val.jsonl",
        "description": "FEVER binary claim verification (true/false).",
    },
}


def fever_rows(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("paper_dev", "paper_test"):
        labels: dict[str, set[str]] = {}
        first_id: dict[str, int] = {}
        with (raw_dir / "fever" / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                label = item.get("label", "")
                if label not in ("REFUTES", "SUPPORTS"):
                    continue
                claim = item["claim"]
                labels.setdefault(claim, set()).add(label)
                first_id.setdefault(claim, int(item["id"]))
        for claim, claim_labels in labels.items():
            if len(claim_labels) != 1:
                continue
            label = next(iter(claim_labels))
            question = claim.strip()
            if question.endswith(".") or question.endswith("?"):
                prompt = "Consider the following claim: " + question + " Is this claim true or false. The claim is <mask>"
            else:
                prompt = "Consider the following claim: " + question + ". Is this claim true or false. The claim is <mask>"
            rows.append(
                {
                    "id": f"fever-{split}-{first_id[claim]}",
                    "prompt": prompt,
                    "label": 1 if label == "SUPPORTS" else 0,
                    "answer": "true" if label == "SUPPORTS" else "false",
                }
            )
    return rows


def bios_rows(raw_dir: Path) -> list[dict]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    table = pq.read_table(raw_dir / "bias_in_bios" / "dev.parquet", columns=["hard_text", "profession"])
    texts = table.column("hard_text").to_pylist()
    professions = table.column("profession").to_pylist()
    rows: list[dict] = []
    for index, (text, profession) in enumerate(zip(texts, professions)):
        name = BIOS_PROFESSION_IDS[int(profession)]
        if name not in BIOS_OCCUPATIONS:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][-BIOS_MAX_TOKENS:]
        try:
            question = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except TypeError:
            question = tokenizer.decode(token_ids, skip_special_tokens=True)
        if question.strip().endswith(".") or question.strip().endswith("?"):
            prompt = "Consider the following text: " + question.strip()
        else:
            prompt = "Consider the following text: " + question.strip() + "."
        prompt += " The profession of the person in the text is called <mask>."
        if prompt.count("<mask>") != 1:
            continue
        rows.append(
            {
                "id": f"bios-dev-{index}",
                "prompt": prompt,
                "label": BIOS_OCCUPATIONS.index(name),
                "answer": name,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "downloads" / "raw"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    builders = {"bios_profession_roberta": bios_rows, "fever_roberta_binary": fever_rows}
    manifest = {"split": "validation (first 20% of each ordered dataset)", "tasks": {}}
    for name, spec in TASKS.items():
        rows = builders[name](raw_dir)
        n_val = len(rows) // VAL_FRACTION_DENOM
        val = rows[:n_val]
        digest = write_jsonl(out_dir / spec["file"], val)
        counts: dict[str, int] = {}
        for row in val:
            counts[row["answer"]] = counts.get(row["answer"], 0) + 1
        manifest["tasks"][name] = {
            "file": spec["file"],
            "sha256": digest,
            "n_examples": len(val),
            "n_full_dataset": len(rows),
            "candidates": spec["candidates"],
            "description": spec["description"],
            "label_counts": counts,
        }
        print(f"{name}: {len(rows)} examples in full ordered dataset, wrote {len(val)} validation rows {counts}")
    (out_dir / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
