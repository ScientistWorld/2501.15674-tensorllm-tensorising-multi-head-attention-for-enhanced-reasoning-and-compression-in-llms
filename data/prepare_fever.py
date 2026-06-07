#!/usr/bin/env python3
"""Prepare a deterministic binary FEVER subset for offline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABEL_MAP = {"SUPPORTS": "true", "REFUTES": "false"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/downloads/fever/shared_task_dev.jsonl")
    parser.add_argument("--output", default="data/fever_binary_dev.jsonl")
    parser.add_argument("--per-class", type=int, default=160)
    args = parser.parse_args()

    counts = {label: 0 for label in LABEL_MAP.values()}
    rows = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            label = LABEL_MAP.get(item.get("label"))
            if label is None or counts[label] >= args.per_class:
                continue
            rows.append(
                {
                    "id": item["id"],
                    "claim": item["claim"],
                    "label": label,
                }
            )
            counts[label] += 1
            if all(value >= args.per_class for value in counts.values()):
                break

    if not rows or len(set(counts.values())) != 1:
        raise RuntimeError(f"Could not build a balanced subset from {args.input}: {counts}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} FEVER examples to {out} with counts {counts}")


if __name__ == "__main__":
    main()
