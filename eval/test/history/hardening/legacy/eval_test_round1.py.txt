#!/usr/bin/env python3
"""Held-out scorer for post-training attention-weight modifications of RoBERTa.

Every method is a directory ``outputs/submissions/<method>/`` holding, per
experiment, a structured low-rank replacement of attention projection matrices
(``<experiment>/patch.json``; format in ``briefing/evaluation.md``): which
layers and matrices, how they are arranged into a tensor, and the multilinear
rank of each compressed tensor mode.  The submission is structure only:

* it verifies the fixed pretrained model files against pinned SHA-256 hashes;
* it computes every replaced matrix itself: a Tucker/HOOI approximation
  (HOSVD start, <= 100 sweeps, tolerance 1e-8) of the ORIGINAL pretrained
  weights with the declared layout and ranks, and counts the parameters of
  that representation (factor matrices + core + matrices of a modified layer
  left dense);
* it runs the fixed model with the rebuilt weights on held-out prompts that
  exist only under ``eval/test/data/`` and computes accuracy, gold-token
  negative log-likelihood and the compression ratio of the modified layers;
* the unmodified pretrained model (``roberta_original``) is always scored by
  the evaluator itself and fixes the reasoning gate: loss and compression are
  only credited when accuracy beats both the unmodified model and the
  majority-class rate of the split;
* every method directory on disk is scored, and so is every method key that
  appears in ``scoring/scores.json`` or ``scoring/scores_train.json`` -- a key
  without a valid patch is scored as failed;
* anything invalid (missing/unsafe patch, non-finite values, a representation
  that is not smaller than the dense original, non-finite model outputs, a
  crash) scores the method as failed: the worst value in every metric of that
  experiment.

Usage: ``python eval/test/evaluate.py`` (no arguments needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLICE = "test"
DATA_DIR = ROOT / "eval" / "test" / "data"
DATA_MANIFEST = DATA_DIR / "MANIFEST.json"
MODEL_DIR = ROOT / "data" / "downloads" / "roberta-base"
DEFAULT_SUBMISSIONS = ROOT / "outputs" / "submissions"
DEFAULT_OUTPUT = ROOT / "scoring" / "scores_test.json"
DEFAULT_KEY_SOURCES = [ROOT / "scoring" / "scores.json", ROOT / "scoring" / "scores_train.json"]

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
CR_CAP = 64.0  # compression credit is capped here
WORST = {"accuracy": 0.0, "loss": LOSS_CAP, "mha_compression_ratio": 1.0}

N_LAYERS, D_MODEL, N_HEADS, HEAD_DIM = 12, 768, 12, 64
MATRICES = ("query", "key", "value", "output")
PAD_ID, MASK_ID = 1, 50264
LN_EPS = 1e-5

PATCH_FORMAT = "mha-structure-patch-v3"
MAX_BLOCKS = 4 * N_LAYERS


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
# Patch materialisation: a submission declares structure only (layer, target
# matrices, tensor layout, multilinear ranks).  The evaluator computes the
# decomposition itself -- HOOI (HOSVD initialisation, at most 100 sweeps,
# tolerance 1e-8 on the change of relative error, as TensorLy's
# partial_tucker) of the ORIGINAL pretrained weights on the declared modes --
# rebuilds the weights from it and counts the parameters (factors + core +
# matrices of a modified layer left dense).
# --------------------------------------------------------------------------

LAYOUT_SHAPES = {
    # targets stacked as a d_model x (h * d_v) x n_targets tensor
    "stack": lambda n: (D_MODEL, N_HEADS * HEAD_DIM, n),
    # multi-head arrangement: d_model x d_v x n_targets x h
    "heads": lambda n: (D_MODEL, HEAD_DIM, n, N_HEADS),
}
HOOI_MAX_ITER = 100
HOOI_TOL = 1e-8


def to_layout(mats, layout: str):
    """[n, d_model, h*d_v] canonical matrices -> layout tensor."""
    n = mats.shape[0]
    if layout == "stack":
        return mats.permute(1, 2, 0)
    return mats.reshape(n, D_MODEL, N_HEADS, HEAD_DIM).permute(1, 3, 0, 2)


def from_layout(tensor, layout: str, n: int):
    if layout == "stack":
        return tensor.permute(2, 0, 1)
    return tensor.permute(2, 0, 3, 1).reshape(n, D_MODEL, N_HEADS * HEAD_DIM)


def unfold(tensor, mode: int):
    return tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor, matrix, mode: int):
    """tensor x_mode matrix, matrix of shape [J, I_mode]."""
    moved = tensor.movedim(mode, 0)
    shape = moved.shape
    out = matrix @ moved.reshape(shape[0], -1)
    return out.reshape((matrix.shape[0],) + tuple(shape[1:])).movedim(0, mode)


def leading_left_singular_vectors(matrix, rank: int):
    import torch

    _, vectors = torch.linalg.eigh(matrix @ matrix.T)
    return vectors[:, -rank:].flip(-1).contiguous()


def hooi_reconstruction(tensor, ranks: dict[int, int]):
    """Partial-Tucker HOOI of ``tensor`` (float64) on the modes in ``ranks``; returns the rebuilt tensor."""
    modes = sorted(ranks)
    norm_sq = float((tensor * tensor).sum())
    factors = {m: leading_left_singular_vectors(unfold(tensor, m), ranks[m]) for m in modes}
    errors: list[float] = []
    for iteration in range(HOOI_MAX_ITER):
        for mode in modes:
            projected = tensor
            for other in modes:
                if other != mode:
                    projected = mode_dot(projected, factors[other].T, other)
            factors[mode] = leading_left_singular_vectors(unfold(projected, mode), ranks[mode])
        core = tensor
        for mode in modes:
            core = mode_dot(core, factors[mode].T, mode)
        errors.append(math.sqrt(abs(norm_sq - float((core * core).sum()))) / math.sqrt(norm_sq))
        if iteration > 1 and abs(errors[-2] - errors[-1]) < HOOI_TOL:
            break
    core = tensor
    for mode in modes:
        core = mode_dot(core, factors[mode].T, mode)
    rebuilt = core
    for mode in modes:
        rebuilt = mode_dot(rebuilt, factors[mode], mode)
    return rebuilt, errors[-1]


def materialize_patch(patch_dir: Path, model):
    """Validate a patch and compute its replaced matrices; raise PatchError if invalid."""
    import torch

    spec_path = patch_dir / "patch.json"
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

    overrides: dict[tuple[int, str], "torch.Tensor"] = {}
    touched: set[int] = set()
    stored_params = 0
    errors = []
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
        for target in targets:
            if (layer, target) in overrides:
                raise PatchError(f"block {index}: layer {layer} {target} is replaced twice")
        layout = block.get("layout")
        if layout not in LAYOUT_SHAPES:
            raise PatchError(f"block {index}: layout must be one of {sorted(LAYOUT_SHAPES)}")
        shape = LAYOUT_SHAPES[layout](len(targets))
        raw_ranks = block.get("ranks")
        if not isinstance(raw_ranks, dict) or not raw_ranks:
            raise PatchError(f"block {index}: 'ranks' must map at least one mode index to a rank")
        ranks: dict[int, int] = {}
        for mode_key, rank in raw_ranks.items():
            if not (isinstance(mode_key, str) and mode_key.isascii() and mode_key.isdigit()
                    and int(mode_key) < len(shape)):
                raise PatchError(f"block {index}: rank keys must be mode indices 0..{len(shape) - 1}")
            mode = int(mode_key)
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= shape[mode]:
                raise PatchError(f"block {index}: rank of mode {mode} must be an integer in [1, {shape[mode]}]")
            ranks[mode] = rank
        kept = [ranks.get(m, shape[m]) for m in range(len(shape))]
        for mode, rank in ranks.items():
            if rank > math.prod(kept[m] for m in range(len(shape)) if m != mode):
                raise PatchError(f"block {index}: rank {rank} of mode {mode} exceeds the product of the other ranks")
        stored_params += sum(shape[m] * r for m, r in ranks.items()) + math.prod(kept)
        original = torch.stack([model.canonical_matrix(layer, target) for target in targets]).double()
        rebuilt, rel_error = hooi_reconstruction(to_layout(original, layout).contiguous(), ranks)
        errors.append(rel_error)
        mats = from_layout(rebuilt, layout, len(targets)).to(torch.float32)
        if not bool(torch.isfinite(mats).all()):
            raise PatchError(f"block {index}: rebuilt weights are non-finite")
        for position, target in enumerate(targets):
            overrides[(layer, target)] = mats[position].contiguous()
        touched.add(layer)

    original_params = 4 * D_MODEL * D_MODEL * len(touched)
    kept_dense = D_MODEL * D_MODEL * sum(
        1 for layer in touched for matrix in MATRICES if (layer, matrix) not in overrides
    )
    represented = stored_params + kept_dense
    if touched:
        if represented >= original_params:
            raise PatchError(
                f"constraint violated: modified layers need {represented} parameters, "
                f"not fewer than their dense original ({original_params})"
            )
        ratio = original_params / represented
    else:
        ratio = 1.0
    info = {
        "compression_ratio": float(ratio),
        "stored_params": int(stored_params),
        "represented_params": int(represented),
        "original_params": int(original_params),
        "relative_errors": errors,
        "touched_layers": sorted(touched),
        "replaced": sorted(f"{layer}.{matrix}" for layer, matrix in overrides),
    }
    return overrides, info


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

    def canonical_matrix(self, layer: int, matrix: str):
        """Pretrained weight in the canonical (d_model) x (head-major h*d_v) layout."""
        weight = self.layers[layer][matrix]
        return weight if matrix == "output" else weight.t()

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
            size = max(1, token_budget // max(len(encoded[order[start]]), 1))
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
        labels = [int(row["label"]) for row in rows]
        counts = [labels.count(c) for c in range(len(candidates))]
        prepared[name] = {
            "batches": batches,
            "candidates": torch.tensor(candidates, device=device),
            "labels": labels,
            "ids": [row["id"] for row in rows],
            "majority_accuracy": 100.0 * max(counts) / len(labels),
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


def summarize(record):
    n = len(record["correct"])
    nll = [min(value, LOSS_CAP) if math.isfinite(value) else LOSS_CAP for value in record["nll"]]
    return {"accuracy": 100.0 * sum(record["correct"]) / n, "loss": sum(nll) / n}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def discover_methods(root: Path, key_sources: list[Path]) -> tuple[list[str], dict[str, str]]:
    """Every method directory on disk plus every key named in the score files."""
    found: dict[str, str] = {}
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.name.startswith((".", "_")) or not entry.is_dir() or entry.name in RESERVED_KEYS:
                continue
            found[entry.name] = "directory"
    for source in key_sources:
        try:
            data = json.loads(Path(source).read_text())
            experiments = data.get("experiments", data)
            for name in EXPERIMENTS:
                results = experiments.get(name, {})
                results = results.get("results", results) if isinstance(results, dict) else {}
                for key in results:
                    if isinstance(key, str) and not key.startswith("_") and key not in RESERVED_KEYS:
                        found.setdefault(key, f"listed in {Path(source).name}")
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(found), found


def configure_torch():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    except AttributeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        # Plain fp32 attention kernel: no reduced-precision or fused shortcuts.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", default=str(DEFAULT_SUBMISSIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--key-sources", nargs="*", default=[str(p) for p in DEFAULT_KEY_SOURCES],
                        help="score files whose method keys must also be scored")
    parser.add_argument("--dump-dir", default=None, help="optional per-example outputs for analysis")
    parser.add_argument("--debug-limit", type=int, default=None, help="debug only: score a prefix of each split")
    args = parser.parse_args()

    started = time.time()
    submissions = Path(args.submissions)
    methods, origins = discover_methods(submissions, [Path(p) for p in args.key_sources])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()  # a stale score file never survives a run

    device = configure_torch()
    keys = [BASELINE_KEY] + methods
    status = {key: {} for key in keys}
    results = {name: {} for name in EXPERIMENTS}
    gates = {}

    def fail_all(reason: str) -> None:
        for name in EXPERIMENTS:
            for key in keys:
                results[name][key] = dict(WORST)
                status[key][name] = {"status": "failed", "reason": reason}

    problems = verify_model_files()
    prepared = None
    if problems:
        fail_all("fixed model altered: " + "; ".join(problems))
    else:
        try:
            examples = load_examples(args.debug_limit)
            budget = 32768 if device.type == "cuda" else 4096
            prepared = tokenize(examples, device, budget)
        except Exception as exc:  # noqa: BLE001
            problems = [f"evaluation data unusable: {exc}"]
            fail_all(problems[0])
    if prepared is not None:
        model = FixedRoberta(device)
        for name in EXPERIMENTS:
            runnable = {BASELINE_KEY: {}}
            info_by_key = {BASELINE_KEY: {"compression_ratio": 1.0, "touched_layers": [], "stored_params": 0}}
            for key in methods:
                patch_dir = submissions / key / name
                try:
                    if not patch_dir.is_dir():
                        raise PatchError(f"no patch for experiment {name} ({origins[key]})")
                    overrides, info = materialize_patch(patch_dir, model)
                except PatchError as exc:
                    results[name][key] = dict(WORST)
                    status[key][name] = {"status": "failed", "reason": str(exc)}
                    continue
                except Exception as exc:  # noqa: BLE001
                    results[name][key] = dict(WORST)
                    status[key][name] = {"status": "failed", "reason": f"evaluator error: {exc!r}"}
                    continue
                by_layer: dict[int, dict] = {}
                for (layer, matrix), value in overrides.items():
                    by_layer.setdefault(layer, {})[matrix] = canonical_to_module(matrix, value).to(device)
                runnable[key] = by_layer
                info_by_key[key] = info
            try:
                records = score_experiment(model, prepared[name], runnable)
            except Exception:  # noqa: BLE001  (one bad method must not sink the others)
                print(traceback.format_exc(), file=sys.stderr)
                records = {}
                for key in runnable:
                    try:
                        records.update(score_experiment(model, prepared[name], {key: runnable[key]}))
                    except Exception as exc:  # noqa: BLE001
                        records[key] = {"error": repr(exc)}
            base = records.get(BASELINE_KEY, {})
            if "error" in base or not base.get("finite", False):
                for key in keys:
                    results[name][key] = dict(WORST)
                    status[key][name] = {"status": "failed", "reason": "the unmodified model could not be scored"}
                runnable.clear()
                continue
            base_metrics = summarize(base)
            threshold = max(base_metrics["accuracy"], prepared[name]["majority_accuracy"])
            gates[name] = {
                "baseline_accuracy": base_metrics["accuracy"],
                "baseline_loss": base_metrics["loss"],
                "majority_class_accuracy": prepared[name]["majority_accuracy"],
                "accuracy_threshold": threshold,
                "n_examples": len(prepared[name]["labels"]),
            }
            for key, record in records.items():
                info = info_by_key[key]
                if "error" in record or not record["finite"]:
                    results[name][key] = dict(WORST)
                    reason = record.get("error", "model outputs became non-finite")
                    status[key][name] = {"status": "failed", "reason": reason}
                    continue
                raw = summarize(record)
                raw_ratio = float(info["compression_ratio"])
                if key == BASELINE_KEY:
                    scored = {**raw, "mha_compression_ratio": 1.0}
                    passed = None
                else:
                    passed = raw["accuracy"] > threshold + 1e-9
                    scored = {
                        "accuracy": raw["accuracy"],
                        "loss": raw["loss"] if passed else max(raw["loss"], base_metrics["loss"]),
                        "mha_compression_ratio": min(raw_ratio, CR_CAP) if passed else 1.0,
                    }
                results[name][key] = scored
                status[key][name] = {
                    "status": "ok",
                    "reasoning_gate_passed": passed,
                    "raw_accuracy": raw["accuracy"],
                    "raw_loss": raw["loss"],
                    "raw_compression_ratio": raw_ratio,
                    "touched_layers": info.get("touched_layers", []),
                    "stored_params": info.get("stored_params", 0),
                }
                if args.dump_dir:
                    dump = Path(args.dump_dir) / name
                    dump.mkdir(parents=True, exist_ok=True)
                    (dump / f"{key}.json").write_text(
                        json.dumps({"correct": record["correct"], "nll": record["nll"], "pred": record["pred"]})
                    )
            runnable.clear()

    scores = {
        "metadata": {
            "slice": SLICE,
            "submissions": str(submissions),
            "device": str(device),
            "fixed_model_verified": not problems,
            "debug_limit": args.debug_limit,
            "seconds": round(time.time() - started, 1),
            "gates": gates,
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
