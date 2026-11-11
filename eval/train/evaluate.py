#!/usr/bin/env python3
"""Visible-split scorer for post-training attention-weight modifications of RoBERTa.

This is the public twin of the held-out scorer: identical interface, model,
metrics and failure rules, applied to the public validation split in
``data/tasks/``.

Every method is a directory ``outputs/submissions/<method>/`` holding, per
experiment, a factorized replacement of attention projection matrices
(``<experiment>/patch.json`` + ``<experiment>/tensors.safetensors``; format in
``briefing/evaluation.md``).  This scorer trusts nothing a method writes except
those tensors:

* it verifies the fixed pretrained model files against pinned SHA-256 hashes;
* it rebuilds every replaced matrix from the stored factors (in an isolated
  subprocess) and counts the parameters itself;
* it runs the fixed model with the rebuilt weights on the public validation
  prompts in ``data/tasks/`` and computes accuracy, gold-token
  negative log-likelihood and the compression ratio of the modified layers;
* the unmodified pretrained model (``roberta_original``) is always scored by
  the evaluator itself;
* anything invalid (missing/unsafe patch, non-finite values, a representation
  larger than the dense original, non-finite model outputs) scores the method
  as failed: the worst value in every metric of that experiment.

Usage: ``python eval/train/evaluate.py`` (no arguments needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLICE = "train"
DATA_DIR = ROOT / "data" / "tasks"
DATA_MANIFEST = DATA_DIR / "tasks.json"
MODEL_DIR = ROOT / "data" / "downloads" / "roberta-base"
DEFAULT_SUBMISSIONS = ROOT / "outputs" / "submissions"
DEFAULT_OUTPUT = ROOT / "scoring" / "scores_train.json"

BASELINE_KEY = "roberta_original"
RESERVED_KEYS = {BASELINE_KEY}

# roberta-base at revision e2da8e2f811d1448a5b465c236feacd80ffbac7b
MODEL_SHA256 = {
    "config.json": "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "vocab.json": "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
}

EXPERIMENTS = {
    "bios_profession_roberta": [
        " journalist", " poet", " composer", " model", " teacher", " architect", " painter", " professor",
    ],
    "fever_roberta_binary": [" false", " true"],
}

METRICS = ("accuracy", "loss", "mha_compression_ratio")
LOSS_CAP = 30.0  # nats; per-example gold NLL is clipped here, failed runs get this mean
WORST = {"accuracy": 0.0, "loss": LOSS_CAP, "mha_compression_ratio": 1.0}

N_LAYERS, D_MODEL, N_HEADS, HEAD_DIM = 12, 768, 12, 64
MATRICES = ("query", "key", "value", "output")
PAD_ID, MASK_ID = 1, 50264
LN_EPS = 1e-5

PATCH_FORMAT = "mha-factor-patch-v1"
MAX_BLOCKS = 4 * N_LAYERS
MAX_TENSORS = 512
MAX_TENSOR_FILE_BYTES = 512 * 1024 * 1024
MAX_EQUATION_CHARS = 256
EQUATION_RE = re.compile(r"^[A-Za-z]+(,[A-Za-z]+)*->[A-Za-z]*$")
METHOD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class PatchError(Exception):
    """Raised when a submission cannot be scored as a valid modification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files() -> list[str]:
    problems = []
    for name, expected in MODEL_SHA256.items():
        path = MODEL_DIR / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != expected:
            problems.append(f"{name} does not match the pinned roberta-base checkpoint")
    return problems


# --------------------------------------------------------------------------
# Patch materialisation (runs in a separate process for isolation)
# --------------------------------------------------------------------------


def materialize_patch(patch_dir: Path):
    import torch

    spec_path = patch_dir / "patch.json"
    tensor_path = patch_dir / "tensors.safetensors"
    if not spec_path.is_file():
        raise PatchError("patch.json not found")
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, ValueError) as exc:
        raise PatchError(f"patch.json unreadable: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("format") != PATCH_FORMAT:
        raise PatchError(f"patch.json must be an object with format={PATCH_FORMAT!r}")
    blocks = spec.get("blocks")
    if not isinstance(blocks, list) or len(blocks) > MAX_BLOCKS:
        raise PatchError(f"'blocks' must be a list with at most {MAX_BLOCKS} entries")

    tensors: dict[str, "torch.Tensor"] = {}
    if blocks:
        if not tensor_path.is_file():
            raise PatchError("tensors.safetensors not found")
        if tensor_path.stat().st_size > MAX_TENSOR_FILE_BYTES:
            raise PatchError("tensors.safetensors is too large")
        from safetensors.torch import load_file

        try:
            raw = load_file(str(tensor_path), device="cpu")
        except Exception as exc:  # noqa: BLE001
            raise PatchError(f"tensors.safetensors unreadable: {exc}") from exc
        if len(raw) > MAX_TENSORS:
            raise PatchError("too many tensors")
        for name, value in raw.items():
            if value.is_complex():
                raise PatchError(f"tensor {name} is complex")
            value = value.to(torch.float32)
            if not bool(torch.isfinite(value).all()):
                raise PatchError(f"tensor {name} contains non-finite values")
            tensors[name] = value
    stored_params = int(sum(value.numel() for value in tensors.values()))

    overrides: dict[tuple[int, str], "torch.Tensor"] = {}
    touched: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise PatchError(f"block {index} is not an object")
        layer = block.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < N_LAYERS:
            raise PatchError(f"block {index}: layer must be an integer in [0, {N_LAYERS - 1}]")
        targets = block.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(target not in MATRICES for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise PatchError(f"block {index}: targets must be distinct names from {MATRICES}")
        equation = block.get("equation")
        if not isinstance(equation, str) or len(equation) > MAX_EQUATION_CHARS:
            raise PatchError(f"block {index}: equation must be a short string")
        equation = equation.replace(" ", "")
        if not EQUATION_RE.match(equation):
            raise PatchError(f"block {index}: equation must look like 'ab,bc->ac' (explicit output, no ellipsis)")
        operands = block.get("operands")
        n_terms = len(equation.split("->")[0].split(","))
        if not isinstance(operands, list) or len(operands) != n_terms:
            raise PatchError(f"block {index}: need one operand name per equation input")
        if any(not isinstance(name, str) or name not in tensors for name in operands):
            raise PatchError(f"block {index}: operands must name tensors stored in tensors.safetensors")
        try:
            with torch.no_grad():
                value = torch.einsum(equation, *[tensors[name].double() for name in operands])
        except Exception as exc:  # noqa: BLE001
            raise PatchError(f"block {index}: contraction failed: {exc}") from exc
        if value.numel() != len(targets) * D_MODEL * D_MODEL:
            raise PatchError(
                f"block {index}: contraction has {value.numel()} entries, expected "
                f"{len(targets)} x {D_MODEL} x {D_MODEL}"
            )
        value = value.reshape(len(targets), D_MODEL, D_MODEL).to(torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise PatchError(f"block {index}: rebuilt weights are non-finite")
        for position, target in enumerate(targets):
            key = (layer, target)
            if key in overrides:
                raise PatchError(f"block {index}: layer {layer} {target} is replaced twice")
            overrides[key] = value[position].contiguous()
        touched.add(layer)

    original_params = 4 * D_MODEL * D_MODEL * len(touched)
    kept_dense = D_MODEL * D_MODEL * sum(
        1 for layer in touched for matrix in MATRICES if (layer, matrix) not in overrides
    )
    if touched:
        represented = stored_params + kept_dense
        if represented <= 0:
            raise PatchError("empty representation")
        if represented > original_params:
            raise PatchError(
                f"constraint violated: modified layers need {represented} parameters, "
                f"more than their dense original ({original_params})"
            )
        ratio = original_params / represented
    else:
        represented = 0
        ratio = 1.0
    info = {
        "compression_ratio": float(ratio),
        "stored_params": stored_params,
        "represented_params": int(represented),
        "original_params": int(original_params),
        "touched_layers": sorted(touched),
        "replaced": sorted(f"{layer}.{matrix}" for layer, matrix in overrides),
    }
    return overrides, info


def materialize_main(patch_dir: Path, out_prefix: Path) -> int:
    import torch  # noqa: F401
    from safetensors.torch import save_file

    try:
        overrides, info = materialize_patch(patch_dir)
    except PatchError as exc:
        out_prefix.with_suffix(".json").write_text(json.dumps({"ok": False, "reason": str(exc)}))
        return 0
    save_file({f"{layer}.{matrix}": value for (layer, matrix), value in overrides.items()},
              str(out_prefix.with_suffix(".safetensors")))
    out_prefix.with_suffix(".json").write_text(json.dumps({"ok": True, **info}))
    return 0


def run_materialize(patch_dir: Path, work: Path, tag: str, timeout: float):
    """Rebuild a patch in a subprocess; return (overrides, info) or raise PatchError."""
    from safetensors.torch import load_file

    prefix = work / tag
    command = [sys.executable, str(Path(__file__).resolve()), "--materialize", str(patch_dir), "--out", str(prefix)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise PatchError("rebuilding the patch timed out") from exc
    meta_path = prefix.with_suffix(".json")
    if proc.returncode != 0 or not meta_path.is_file():
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise PatchError(f"rebuilding the patch crashed: {tail[0][:300]}")
    meta = json.loads(meta_path.read_text())
    if not meta.get("ok"):
        raise PatchError(meta.get("reason", "invalid patch"))
    tensors = load_file(str(prefix.with_suffix(".safetensors"))) if meta["replaced"] else {}
    overrides = {}
    for key, value in tensors.items():
        layer, matrix = key.split(".")
        overrides[(int(layer), matrix)] = value
    meta.pop("ok", None)
    return overrides, meta


# --------------------------------------------------------------------------
# Fixed pretrained model (RoBERTa-base masked LM), evaluated directly
# --------------------------------------------------------------------------


class FixedRoberta:
    def __init__(self, device):
        import torch
        from safetensors.torch import load_file

        state = load_file(str(MODEL_DIR / "model.safetensors"), device="cpu")
        self.device = device

        def get(name):
            return state[name].to(device=device, dtype=torch.float32).contiguous()

        prefix = "roberta.embeddings."
        self.word = get(prefix + "word_embeddings.weight")
        self.position = get(prefix + "position_embeddings.weight")
        self.token_type = get(prefix + "token_type_embeddings.weight")[0]
        self.emb_ln = (get(prefix + "LayerNorm.weight"), get(prefix + "LayerNorm.bias"))
        self.layers = []
        for layer in range(N_LAYERS):
            p = f"roberta.encoder.layer.{layer}."
            self.layers.append(
                {
                    "query": get(p + "attention.self.query.weight"),
                    "query_b": get(p + "attention.self.query.bias"),
                    "key": get(p + "attention.self.key.weight"),
                    "key_b": get(p + "attention.self.key.bias"),
                    "value": get(p + "attention.self.value.weight"),
                    "value_b": get(p + "attention.self.value.bias"),
                    "output": get(p + "attention.output.dense.weight"),
                    "output_b": get(p + "attention.output.dense.bias"),
                    "ln1": (get(p + "attention.output.LayerNorm.weight"), get(p + "attention.output.LayerNorm.bias")),
                    "ffn_in": (get(p + "intermediate.dense.weight"), get(p + "intermediate.dense.bias")),
                    "ffn_out": (get(p + "output.dense.weight"), get(p + "output.dense.bias")),
                    "ln2": (get(p + "output.LayerNorm.weight"), get(p + "output.LayerNorm.bias")),
                }
            )
        self.head_dense = (get("lm_head.dense.weight"), get("lm_head.dense.bias"))
        self.head_ln = (get("lm_head.layer_norm.weight"), get("lm_head.layer_norm.bias"))
        self.head_bias = get("lm_head.bias")

    def embed(self, input_ids):
        import torch
        import torch.nn.functional as F

        not_pad = input_ids.ne(PAD_ID).int()
        positions = (torch.cumsum(not_pad, dim=1).type_as(not_pad) * not_pad).long() + PAD_ID
        hidden = self.word[input_ids] + self.position[positions] + self.token_type
        return F.layer_norm(hidden, (D_MODEL,), self.emb_ln[0], self.emb_ln[1], LN_EPS)

    def layer(self, hidden, keep, index, override=None):
        import torch.nn.functional as F

        params = self.layers[index]
        override = override or {}
        batch, length, _ = hidden.shape

        def proj(name):
            weight = override.get(name, params[name])
            out = F.linear(hidden, weight, params[name + "_b"])
            return out.view(batch, length, N_HEADS, HEAD_DIM).transpose(1, 2)

        context = F.scaled_dot_product_attention(proj("query"), proj("key"), proj("value"), attn_mask=keep)
        context = context.transpose(1, 2).reshape(batch, length, D_MODEL)
        attended = F.linear(context, override.get("output", params["output"]), params["output_b"])
        hidden = F.layer_norm(attended + hidden, (D_MODEL,), params["ln1"][0], params["ln1"][1], LN_EPS)
        inner = F.gelu(F.linear(hidden, *params["ffn_in"]))
        out = F.linear(inner, *params["ffn_out"])
        return F.layer_norm(out + hidden, (D_MODEL,), params["ln2"][0], params["ln2"][1], LN_EPS)

    def mask_log_probs(self, hidden_at_mask):
        import torch.nn.functional as F

        z = F.gelu(F.linear(hidden_at_mask, *self.head_dense))
        z = F.layer_norm(z, (D_MODEL,), self.head_ln[0], self.head_ln[1], LN_EPS)
        logits = F.linear(z, self.word, self.head_bias)
        return F.log_softmax(logits.float(), dim=-1)


def canonical_to_module(matrix: str, value):
    """Canonical layout is (model dim) x (head-major attention dim)."""
    return value if matrix == "output" else value.t()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_examples(limit: int | None):
    manifest = json.loads(DATA_MANIFEST.read_text())
    examples = {}
    for name in EXPERIMENTS:
        entry = manifest["tasks"][name]
        path = DATA_DIR / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"{path} does not match its recorded hash; refusing to score on altered data")
        rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
        if len(rows) != entry["n_examples"]:
            raise RuntimeError(f"{path} has {len(rows)} rows, expected {entry['n_examples']}")
        if limit:
            rows = rows[:limit]
        examples[name] = rows
    return examples


def tokenize(examples, device, token_budget: int):
    import torch
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    prepared = {}
    for name, rows in examples.items():
        candidates = []
        for word in EXPERIMENTS[name]:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            if len(ids) != 1:
                raise RuntimeError(f"candidate {word!r} is not a single token")
            candidates.append(ids[0])
        encoded = [tokenizer.encode(row["prompt"]).ids for row in rows]
        for row, ids in zip(rows, encoded):
            if ids.count(MASK_ID) != 1 or len(ids) > 512:
                raise RuntimeError(f"prompt {row['id']} must contain exactly one mask and fit 512 tokens")
            if not 0 <= int(row["label"]) < len(candidates):
                raise RuntimeError(f"label out of range for {row['id']}")
        order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), i))
        batches = []
        start = 0
        while start < len(order):
            longest = len(encoded[order[start]])
            size = max(1, token_budget // max(longest, 1))
            chunk = order[start : start + size]
            longest = max(len(encoded[i]) for i in chunk)
            ids = torch.full((len(chunk), longest), PAD_ID, dtype=torch.long)
            for row_pos, example in enumerate(chunk):
                ids[row_pos, : len(encoded[example])] = torch.tensor(encoded[example])
            mask_pos = torch.tensor([encoded[i].index(MASK_ID) for i in chunk])
            labels = torch.tensor([int(rows[i]["label"]) for i in chunk])
            batches.append(
                {
                    "index": chunk,
                    "ids": ids.to(device),
                    "keep": ids.ne(PAD_ID)[:, None, None, :].to(device),
                    "mask_pos": mask_pos.to(device),
                    "labels": labels.to(device),
                }
            )
            start += size
        prepared[name] = {
            "batches": batches,
            "candidates": torch.tensor(candidates, device=device),
            "labels": [int(row["label"]) for row in rows],
            "ids": [row["id"] for row in rows],
        }
    return prepared


def score_experiment(model, prepared, methods):
    """Run every method on one experiment, sharing the unmodified layer prefix.

    ``methods`` maps a method key to {layer: {matrix: module-layout weight}}.
    """
    import torch

    n = len(prepared["labels"])
    out = {key: {"correct": [0] * n, "nll": [0.0] * n, "pred": [0] * n, "finite": True} for key in methods}
    starts = {key: (min(layers) if layers else None) for key, layers in methods.items()}
    with torch.inference_mode():
        for batch in prepared["batches"]:
            hidden = model.embed(batch["ids"])
            states = {}
            for layer in range(N_LAYERS):
                for key, start in starts.items():
                    if start == layer:
                        states[key] = hidden
                for key in list(states):
                    states[key] = model.layer(states[key], batch["keep"], layer, methods[key].get(layer))
                hidden = model.layer(hidden, batch["keep"], layer)
            rows = torch.arange(batch["ids"].shape[0], device=batch["ids"].device)
            for key in methods:
                final = states.get(key, hidden)
                log_probs = model.mask_log_probs(final[rows, batch["mask_pos"]])
                cand = log_probs.index_select(1, prepared["candidates"])
                gold = cand.gather(1, batch["labels"][:, None])[:, 0]
                if not (bool(torch.isfinite(cand).all()) and bool(torch.isfinite(gold).all())):
                    out[key]["finite"] = False
                pred = cand.argmax(dim=1).tolist()
                gold_nll = (-gold).double().tolist()
                labels = batch["labels"].tolist()
                record = out[key]
                for pos, example in enumerate(batch["index"]):
                    record["pred"][example] = int(pred[pos])
                    record["correct"][example] = int(pred[pos] == labels[pos])
                    record["nll"][example] = float(gold_nll[pos])
    return out


def summarize(record, labels):
    n = len(labels)
    nll = [min(value, LOSS_CAP) if math.isfinite(value) else LOSS_CAP for value in record["nll"]]
    recalls = []
    for label in sorted(set(labels)):
        members = [i for i in range(n) if labels[i] == label]
        recalls.append(sum(record["correct"][i] for i in members) / len(members))
    return {
        "accuracy": 100.0 * sum(record["correct"]) / n,
        "loss": sum(nll) / n,
    }, {"balanced_accuracy": 100.0 * sum(recalls) / len(recalls), "n_examples": n}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def discover_methods(root: Path) -> tuple[list[str], list[str]]:
    found, skipped = [], []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.name.startswith((".", "_")):
                continue
            if not entry.is_dir():
                continue
            if entry.name in RESERVED_KEYS or not METHOD_NAME_RE.match(entry.name):
                skipped.append(entry.name)
                continue
            found.append(entry.name)
    return found, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", default=str(DEFAULT_SUBMISSIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dump-dir", default=None, help="optional per-example outputs for analysis")
    parser.add_argument("--debug-limit", type=int, default=None, help="debug only: score a prefix of each split")
    parser.add_argument("--patch-timeout", type=float, default=900.0)
    parser.add_argument("--materialize", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.materialize:
        return materialize_main(Path(args.materialize), Path(args.out))

    import torch

    started = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    except AttributeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    methods, skipped = discover_methods(Path(args.submissions))
    keys = [BASELINE_KEY] + methods
    status = {key: {} for key in keys}
    results = {name: {} for name in EXPERIMENTS}

    model_problems = verify_model_files()
    if model_problems:
        for name in EXPERIMENTS:
            for key in keys:
                results[name][key] = dict(WORST)
                status[key][name] = {"status": "failed", "reason": "fixed model altered: " + "; ".join(model_problems)}
    else:
        examples = load_examples(args.debug_limit)
        budget = 16384 if device.type == "cuda" else 4096
        prepared = tokenize(examples, device, budget)
        model = FixedRoberta(device)
        with tempfile.TemporaryDirectory(prefix="mha_eval_") as tmp:
            work = Path(tmp)
            for name in EXPERIMENTS:
                runnable = {BASELINE_KEY: {}}
                info_by_key = {BASELINE_KEY: {"compression_ratio": 1.0, "touched_layers": [], "stored_params": 0}}
                for key in methods:
                    patch_dir = Path(args.submissions) / key / name
                    try:
                        if not patch_dir.is_dir():
                            raise PatchError(f"no patch for experiment {name}")
                        overrides, info = run_materialize(patch_dir, work, f"{key}__{name}", args.patch_timeout)
                    except PatchError as exc:
                        results[name][key] = dict(WORST)
                        status[key][name] = {"status": "failed", "reason": str(exc)}
                        continue
                    by_layer: dict[int, dict] = {}
                    for (layer, matrix), value in overrides.items():
                        by_layer.setdefault(layer, {})[matrix] = canonical_to_module(matrix, value).to(device)
                    runnable[key] = by_layer
                    info_by_key[key] = info
                records = score_experiment(model, prepared[name], runnable)
                for key, record in records.items():
                    info = info_by_key[key]
                    if not record["finite"]:
                        results[name][key] = dict(WORST)
                        status[key][name] = {"status": "failed", "reason": "model outputs became non-finite"}
                        continue
                    metrics, extra = summarize(record, prepared[name]["labels"])
                    metrics["mha_compression_ratio"] = float(info["compression_ratio"])
                    results[name][key] = metrics
                    status[key][name] = {
                        "status": "ok",
                        **extra,
                        "touched_layers": info.get("touched_layers", []),
                        "stored_params": info.get("stored_params", 0),
                    }
                    if args.dump_dir:
                        dump = Path(args.dump_dir) / name
                        dump.mkdir(parents=True, exist_ok=True)
                        (dump / f"{key}.json").write_text(
                            json.dumps({"correct": record["correct"], "nll": record["nll"], "pred": record["pred"]})
                        )
                for key in runnable:
                    runnable[key] = None

    scores = {
        "metadata": {
            "slice": SLICE,
            "submissions": str(Path(args.submissions)),
            "device": str(device),
            "fixed_model_verified": not model_problems,
            "skipped_directories": skipped,
            "debug_limit": args.debug_limit,
            "seconds": round(time.time() - started, 1),
            "methods": status,
        },
        "experiments": {name: {"results": results[name]} for name in EXPERIMENTS},
    }
    tmp_out = output.with_suffix(".tmp")
    tmp_out.write_text(json.dumps(scores, indent=2) + "\n")
    os.replace(tmp_out, output)
    print(json.dumps(scores["experiments"], indent=2))
    for key in keys:
        for name, entry in status[key].items():
            if entry.get("status") != "ok":
                print(f"[failed] {key} / {name}: {entry.get('reason')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
