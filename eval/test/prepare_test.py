#!/usr/bin/env python3
"""Build the hidden held-out evaluation splits.  Lives only in ``eval/test/``.

Same LASER-style preprocessing as the public validation split
(``data/prepare_tasks.py``), applied to data that is never shipped outside
``eval/test/``:

* ``fever_roberta_binary`` -- every consistent support/refute claim of FEVER
  v1.0 ``paper_test`` (the second half of the paper's ordered 13,086-claim
  dataset; it contains the paper's "last 20%" test portion and is used whole
  to shrink the test noise).
* ``bios_profession_roberta`` -- the last 20% of the filtered Bias in Bios
  ``dev`` split (the paper's test portion).

The output files under ``eval/test/data/`` are committed and hash-pinned in
``MANIFEST.json``; the evaluator refuses altered files.  This script is only
needed to rebuild them.  Raw files are read from ``--raw-dir`` (default
``eval/test/raw/``, git-ignored) and fetched there if missing:

    python eval/test/prepare_test.py [--raw-dir eval/test/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "eval" / "test" / "data"
RAW_DIR = ROOT / "eval" / "test" / "raw"
MODEL_DIR = ROOT / "data" / "downloads" / "roberta-base"
VISIBLE_DIR = ROOT / "data" / "tasks"

RAW_FILES = {
    "fever/paper_dev.jsonl": (
        "https://fever.ai/download/fever/paper_dev.jsonl",
        "41158707810008747946bf23471e82df53e77a513524b9e3ec1c2e674ef5ef8c",
    ),
    "fever/paper_test.jsonl": (
        "https://fever.ai/download/fever/paper_test.jsonl",
        "fb7b0280a0adc2302bbb29bfb7af37274fa585de3171bcf908f180642d11d88e",
    ),
    "bias_in_bios/dev.parquet": (
        "https://huggingface.co/datasets/LabHC/bias_in_bios/resolve/main/data/dev-00000-of-00001-e6551072fff26949.parquet",
        "4c3461a19cf437f89fcfea5abbaea43baeb972f706662b3a3811eedce334e3b2",
    ),
}

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
    "bios_profession_roberta": {"file": "bios_profession_test.jsonl", "visible": "bios_profession_val.jsonl"},
    "fever_roberta_binary": {"file": "fever_test.jsonl", "visible": "fever_val.jsonl"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_raw(raw_dir: Path) -> None:
    for rel, (url, digest) in RAW_FILES.items():
        path = raw_dir / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetching {url}")
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise RuntimeError(f"{path} does not match the pinned sha256")


def fever_prompt(claim: str) -> str:
    question = claim.strip()
    if question.endswith(".") or question.endswith("?"):
        return "Consider the following claim: " + question + " Is this claim true or false. The claim is <mask>"
    return "Consider the following claim: " + question + ". Is this claim true or false. The claim is <mask>"


def fever_rows(path: Path, split: str) -> list[dict]:
    labels: dict[str, set[str]] = {}
    first_id: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            label = item.get("label", "")
            if label not in ("REFUTES", "SUPPORTS"):
                continue
            claim = item["claim"]
            labels.setdefault(claim, set()).add(label)
            first_id.setdefault(claim, int(item["id"]))
    rows = []
    for claim, claim_labels in labels.items():
        if len(claim_labels) != 1:
            continue
        label = next(iter(claim_labels))
        rows.append(
            {
                "id": f"fever-{split}-{first_id[claim]}",
                "prompt": fever_prompt(claim),
                "label": 1 if label == "SUPPORTS" else 0,
                "answer": "true" if label == "SUPPORTS" else "false",
            }
        )
    return rows


def laser_cleanup(text: str) -> str:
    """transformers 4.x ``clean_up_tokenization`` (applied by LASER's decode; 5.x skips it for BPE)."""
    return (
        text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
        .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s")
        .replace(" 've", "'ve").replace(" 're", "'re")
    )


def bios_rows(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    table = pq.read_table(parquet, columns=["hard_text", "profession"])
    texts = table.column("hard_text").to_pylist()
    professions = table.column("profession").to_pylist()
    rows: list[dict] = []
    for index, (text, profession) in enumerate(zip(texts, professions)):
        name = BIOS_PROFESSION_IDS[int(profession)]
        if name not in BIOS_OCCUPATIONS:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][-BIOS_MAX_TOKENS:]
        question = laser_cleanup(tokenizer.decode(token_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch_raw(raw_dir)

    fever_dev = fever_rows(raw_dir / "fever" / "paper_dev.jsonl", "paper_dev")
    fever_test = fever_rows(raw_dir / "fever" / "paper_test.jsonl", "paper_test")
    n_fever = len(fever_dev) + len(fever_test)
    bios_all = bios_rows(raw_dir / "bias_in_bios" / "dev.parquet")
    n_bios_test = len(bios_all) // 5
    splits = {
        "fever_roberta_binary": (
            fever_test,
            n_fever,
            "FEVER paper_test portion: positions %d..%d of the ordered dataset" % (len(fever_dev), n_fever - 1),
        ),
        "bios_profession_roberta": (
            bios_all[len(bios_all) - n_bios_test:],
            len(bios_all),
            "last 20%%: positions %d..%d of the ordered dataset" % (len(bios_all) - n_bios_test, len(bios_all) - 1),
        ),
    }

    manifest = {"split": "held-out test", "tasks": {}}
    for name, spec in TASKS.items():
        test, n_full, description = splits[name]
        visible_path = VISIBLE_DIR / spec["visible"]
        if not visible_path.exists():
            raise RuntimeError(f"build the visible split first ({visible_path})")
        visible = [json.loads(line) for line in visible_path.open("r", encoding="utf-8")]
        visible_ids = {row["id"] for row in visible}
        visible_prompts = {row["prompt"] for row in visible}
        if any(row["id"] in visible_ids for row in test):
            raise RuntimeError(f"{name}: held-out ids overlap the visible split")
        # The source data contains a few duplicated texts: a held-out prompt that
        # is also visible (with its label) is dropped.
        dropped = [row["id"] for row in test if row["prompt"] in visible_prompts]
        test = [row for row in test if row["prompt"] not in visible_prompts]
        path = out_dir / spec["file"]
        with path.open("w", encoding="utf-8") as handle:
            for row in test:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts: dict[str, int] = {}
        for row in test:
            counts[row["answer"]] = counts.get(row["answer"], 0) + 1
        manifest["tasks"][name] = {
            "file": spec["file"],
            "sha256": sha256(path),
            "n_examples": len(test),
            "n_full_dataset": n_full,
            "portion": description,
            "dropped_duplicates_of_visible_prompts": dropped,
            "label_counts": counts,
        }
        print(f"{name}: wrote {len(test)} held-out rows {counts}")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
