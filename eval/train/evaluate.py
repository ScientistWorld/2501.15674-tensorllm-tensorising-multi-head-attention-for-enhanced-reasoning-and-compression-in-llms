#!/usr/bin/env python3
"""Train-slice scorer for FEVER prediction artifacts.

This evaluator is intentionally method-agnostic: it reads the manifest and
prediction JSONL files produced by a method run, then scores only the visible
train slice of each artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT = "fever_roberta_binary"
SLICE_NAME = "train"
SLICE_PARITY = 0

DEFAULT_METHOD_MAP = {
    "roberta_original": "roberta_original",
    "tensorllm_best": "tensorllm_roberta",
}


def in_slice(row_index: int) -> bool:
    """Split artifacts into equal visible/held-out halves by row index."""

    return row_index % 2 == SLICE_PARITY


def score_predictions(path: Path) -> tuple[dict[str, float], int]:
    total = 0
    correct = 0
    loss = 0.0
    label_counts: dict[str, int] = {}
    prediction_counts: dict[str, int] = {}
    correct_by_label: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not in_slice(row_index):
                continue
            row = json.loads(line)
            label = str(row["label"])
            prediction = str(row["prediction"])
            total += 1
            label_counts[label] = label_counts.get(label, 0) + 1
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
            correct_by_label.setdefault(label, 0)
            is_correct = prediction == label
            correct += int(is_correct)
            correct_by_label[label] += int(is_correct)
            loss += float(row["nll"])
    if total == 0:
        raise RuntimeError(f"No {SLICE_NAME} predictions found in {path}")
    recalls = [
        correct_by_label.get(label, 0) / count
        for label, count in label_counts.items()
        if count > 0
    ]
    if not recalls:
        raise RuntimeError(f"No labels found in {path}")
    labels = set(label_counts) | set(prediction_counts)
    distribution_shift = 0.5 * sum(
        abs(prediction_counts.get(label, 0) / total - label_counts.get(label, 0) / total)
        for label in labels
    )
    return {
        "accuracy": 100.0 * correct / total,
        "balanced_accuracy": 100.0 * sum(recalls) / len(recalls),
        "loss": loss / total,
        "prediction_distribution_shift": 100.0 * distribution_shift,
    }, total


def score_key(condition: dict) -> str:
    """Return the public score key for a manifest condition."""

    if "score_key" in condition:
        return str(condition["score_key"])
    name = str(condition["name"])
    return DEFAULT_METHOD_MAP.get(name, name)


def resolve_prediction_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def compression_ratio(condition: dict, method_name: str) -> float:
    original = condition.get("original_mha_params")
    compressed = condition.get("compressed_mha_params")
    provided = condition.get("compression_ratio")

    if original is None or compressed is None:
        if provided is None:
            raise RuntimeError(f"{method_name} is missing compression metadata")
        return float(provided)

    if float(compressed) <= 0:
        raise RuntimeError(f"{method_name} has invalid compressed_mha_params={compressed}")
    computed = float(original) / float(compressed)
    if provided is not None and abs(computed - float(provided)) > 1e-6 * max(1.0, computed):
        raise RuntimeError(
            f"{method_name} manifest compression_ratio={provided} does not match "
            f"original_mha_params/compressed_mha_params={computed}"
        )
    return computed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/tensorllm_fever/manifest.json")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    results: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for condition in manifest["conditions"]:
        method_name = score_key(condition)
        if method_name in results:
            raise RuntimeError(f"Duplicate score key in manifest: {method_name}")
        metrics, count = score_predictions(resolve_prediction_path(condition["predictions"], manifest_path))
        metrics["mha_compression_ratio"] = compression_ratio(condition, method_name)
        results[method_name] = metrics
        counts[method_name] = count

    if not results:
        raise RuntimeError("Manifest did not contain any scoreable conditions")

    scores = {
        "metadata": {
            "slice": SLICE_NAME,
            "split": "prediction_row_index_mod_2",
            "manifest": str(manifest_path),
            "examples_per_method": counts,
        },
        "experiments": {EXPERIMENT: {"results": results}},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
