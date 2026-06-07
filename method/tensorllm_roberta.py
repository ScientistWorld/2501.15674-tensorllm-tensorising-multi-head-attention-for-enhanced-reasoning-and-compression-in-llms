#!/usr/bin/env python3
"""TensorLLM reproduction for RoBERTa masked-language-model attention layers."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def mode_unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    order = [mode] + [idx for idx in range(tensor.ndim) if idx != mode]
    return tensor.permute(order).reshape(tensor.shape[mode], -1)


def mode_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    moved = tensor.movedim(mode, 0)
    original_shape = moved.shape
    product = matrix @ moved.reshape(original_shape[0], -1)
    new_shape = (matrix.shape[0],) + tuple(original_shape[1:])
    return product.reshape(new_shape).movedim(0, mode)


def project_except(tensor: torch.Tensor, factors: list[torch.Tensor], skip: int) -> torch.Tensor:
    out = tensor
    for mode, factor in enumerate(factors):
        if mode != skip:
            out = mode_product(out, factor.T, mode)
    return out


def shared_tucker_hooi(tensor: torch.Tensor, ranks: tuple[int, int, int], n_iter: int = 6) -> tuple[torch.Tensor, int]:
    """HOOI Tucker approximation on modes 0, 1, 2, leaving head mode uncompressed."""

    factors: list[torch.Tensor] = []
    for mode, rank in enumerate(ranks):
        u, _, _ = torch.linalg.svd(mode_unfold(tensor, mode), full_matrices=False)
        factors.append(u[:, :rank].contiguous())

    for _ in range(n_iter):
        for mode, rank in enumerate(ranks):
            projected = project_except(tensor, factors, mode)
            u, _, _ = torch.linalg.svd(mode_unfold(projected, mode), full_matrices=False)
            factors[mode] = u[:, :rank].contiguous()

    core = tensor
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor.T, mode)

    recon = core
    for mode, factor in enumerate(factors):
        recon = mode_product(recon, factor, mode)

    compressed_params = (
        tensor.shape[0] * ranks[0]
        + tensor.shape[1] * ranks[1]
        + tensor.shape[2] * ranks[2]
        + ranks[0] * ranks[1] * ranks[2] * tensor.shape[3]
    )
    return recon, compressed_params


def roberta_layer_tensor(layer) -> torch.Tensor:
    self_attn = layer.attention.self
    output = layer.attention.output
    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size
    d_model = self_attn.all_head_size

    q = self_attn.query.weight.detach().T.reshape(d_model, num_heads, head_dim)
    k = self_attn.key.weight.detach().T.reshape(d_model, num_heads, head_dim)
    v = self_attn.value.weight.detach().T.reshape(d_model, num_heads, head_dim)
    # PyTorch stores the output projection as W^T. This already has the
    # paper's W^O^T orientation: d_model x (h * d_v).
    o = output.dense.weight.detach().reshape(d_model, num_heads, head_dim)
    return torch.stack([q, k, v, o], dim=2).permute(0, 3, 2, 1).contiguous()


def write_roberta_layer_tensor(layer, tensor: torch.Tensor) -> None:
    self_attn = layer.attention.self
    output = layer.attention.output
    d_model, head_dim, four, num_heads = tensor.shape
    if four != 4:
        raise ValueError(f"Expected four MHA matrices, got {four}")
    by_matrix = tensor.permute(0, 3, 1, 2).contiguous()
    q, k, v, o = [by_matrix[..., idx].reshape(d_model, num_heads * head_dim) for idx in range(4)]
    self_attn.query.weight.data.copy_(q.T)
    self_attn.key.weight.data.copy_(k.T)
    self_attn.value.weight.data.copy_(v.T)
    output.dense.weight.data.copy_(o)


def apply_tensorllm(model, layer_idx: int, ranks: tuple[int, int, int], n_iter: int) -> dict:
    layer = model.roberta.encoder.layer[layer_idx]
    tensor = roberta_layer_tensor(layer).to(dtype=torch.float32)
    original_params = int(math.prod(tensor.shape))
    recon, compressed_params = shared_tucker_hooi(tensor, ranks, n_iter=n_iter)
    write_roberta_layer_tensor(layer, recon.to(next(model.parameters()).dtype))
    return {
        "layer": layer_idx,
        "ranks": list(ranks),
        "original_mha_params": original_params,
        "compressed_mha_params": int(compressed_params),
        "compression_ratio": float(original_params / compressed_params),
        "relative_reconstruction_error": float(torch.linalg.norm(tensor - recon) / torch.linalg.norm(tensor)),
    }


def load_records(path: str, limit: int | None) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


@torch.no_grad()
def evaluate_fever(model, tokenizer, rows: list[dict], batch_size: int, device: torch.device) -> list[dict]:
    model.eval()
    true_token = tokenizer.encode(" true", add_special_tokens=False)
    false_token = tokenizer.encode(" false", add_special_tokens=False)
    if len(true_token) != 1 or len(false_token) != 1:
        true_token = tokenizer.encode("true", add_special_tokens=False)
        false_token = tokenizer.encode("false", add_special_tokens=False)
    if len(true_token) != 1 or len(false_token) != 1:
        raise RuntimeError("Could not map true/false labels to single RoBERTa tokens")
    label_to_token = {"true": true_token[0], "false": false_token[0]}
    candidate_ids = torch.tensor([label_to_token["false"], label_to_token["true"]], device=device)

    outputs = []
    for batch in tqdm(list(batched(rows, batch_size)), desc="Evaluating"):
        prompts = [
            f"Consider the following claim: {row['claim']}. Is this claim true or false? The claim is {tokenizer.mask_token}."
            for row in batch
        ]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
        logits = model(**encoded).logits
        mask_positions = encoded["input_ids"].eq(tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.shape[0] != len(batch):
            raise RuntimeError("Each prompt must contain exactly one mask token")
        mask_logits = logits[mask_positions[:, 0], mask_positions[:, 1], :]
        binary_logits = mask_logits.index_select(dim=1, index=candidate_ids)
        log_probs = F.log_softmax(binary_logits, dim=-1)
        pred_idx = torch.argmax(log_probs, dim=-1).cpu().tolist()
        for row, idx, lp in zip(batch, pred_idx, log_probs.cpu()):
            pred = "true" if idx == 1 else "false"
            gold_idx = 1 if row["label"] == "true" else 0
            outputs.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": pred,
                    "correct": pred == row["label"],
                    "nll": float(-lp[gold_idx]),
                    "claim": row["claim"],
                }
            )
    return outputs


def run_condition(args, tokenizer, base_model, rows: list[dict], name: str, layer: int | None, ranks: tuple[int, int, int] | None) -> dict:
    device = torch.device(args.device)
    model = copy.deepcopy(base_model).to(device)
    compression = {
        "layer": None,
        "ranks": None,
        "original_mha_params": None,
        "compressed_mha_params": None,
        "compression_ratio": 1.0,
        "relative_reconstruction_error": 0.0,
    }
    if layer is not None and ranks is not None:
        compression = apply_tensorllm(model, layer, ranks, args.hooi_iters)
        model.to(device)
    predictions = evaluate_fever(model, tokenizer, rows, args.batch_size, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{name}.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item) + "\n")
    acc = sum(item["correct"] for item in predictions) / len(predictions)
    loss = sum(item["nll"] for item in predictions) / len(predictions)
    return {
        "name": name,
        "predictions": str(pred_path),
        "accuracy": acc,
        "loss": loss,
        **compression,
    }


def parse_rank(text: str) -> tuple[int, int, int]:
    vals = tuple(int(part) for part in text.split(","))
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("rank must be R1,R2,R3")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="data/downloads/roberta-base")
    parser.add_argument("--data", default="data/fever_binary_dev.jsonl")
    parser.add_argument("--output-dir", default="outputs/tensorllm_fever")
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--ranks", nargs="+", type=parse_rank, default=[(384, 32, 3), (256, 32, 3), (192, 24, 3)])
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hooi-iters", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_records(args.data, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    base_model = AutoModelForMaskedLM.from_pretrained(args.model_path, local_files_only=True)

    manifest = {
        "paper": "TensorLLM",
        "model": args.model_path,
        "dataset": args.data,
        "task": "fever_binary_masked_lm",
        "conditions": [],
    }
    manifest["conditions"].append(run_condition(args, tokenizer, base_model, rows, "roberta_original", None, None))

    best = None
    for layer in [int(x) for x in args.layers.split(",") if x.strip()]:
        for ranks in args.ranks:
            name = f"tensorllm_layer{layer}_r{'-'.join(map(str, ranks))}"
            result = run_condition(args, tokenizer, base_model, rows, name, layer, ranks)
            manifest["conditions"].append(result)
            if best is None or result["accuracy"] > best["accuracy"] or (
                result["accuracy"] == best["accuracy"] and result["loss"] < best["loss"]
            ):
                best = result

    if best is not None:
        best_copy = dict(best)
        best_copy["name"] = "tensorllm_best"
        manifest["conditions"].append(best_copy)

    manifest_path = Path(args.output_dir) / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"manifest": str(manifest_path), "best": best}, indent=2))


if __name__ == "__main__":
    main()
