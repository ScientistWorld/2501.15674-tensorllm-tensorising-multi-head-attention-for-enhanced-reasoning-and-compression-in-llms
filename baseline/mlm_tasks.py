"""Shared utilities for methods that modify RoBERTa attention weights.

This module is method-agnostic infrastructure for producing submissions:

* loading the public validation splits in ``data/tasks/``: multiple-choice
  tasks answered at one masked position, and open-ended tasks
  (``"type": "generation"`` in ``tasks.json``) answered at appended mask
  positions with the same accuracy / loss definitions as the scorers;
* a direct forward pass of the fixed ``roberta-base`` masked LM (same maths as
  Hugging Face ``RobertaForMaskedLM``) with optional per-layer attention-weight
  overrides and cached hidden states below the first modified layer, so that
  layer-wise searches are cheap;
* the canonical attention-matrix layout used by the submission format and a
  writer for ``patch.json``.  A patch names, per block, a layer, its target
  matrices, optionally a subset of heads, a tensor ``layout`` (``stack``,
  ``heads`` or any grouping of the axes t/m/h/d into modes) and the
  multilinear rank of each compressed mode; the scorers compute the
  Tucker/HOOI approximation of the original weights themselves (see
  briefing/evaluation.md).  ``python eval/train/evaluate.py --submissions DIR``
  scores any candidate patches on the visible split.

It does not import anything from ``eval/``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "downloads" / "roberta-base"
TASK_DIR = ROOT / "data" / "tasks"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"
PATCH_FORMAT = "mha-structure-patch-v3"

N_LAYERS, D_MODEL, N_HEADS, HEAD_DIM = 12, 768, 12, 64
MATRICES = ("query", "key", "value", "output")
PAD_ID, MASK_ID = 1, 50264
LN_EPS = 1e-5
LOSS_CAP = 30.0


def pick_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    except AttributeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        # Same plain fp32 attention kernel the scorers use.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


class Roberta:
    """Fixed roberta-base masked LM with per-layer attention overrides."""

    def __init__(self, device: torch.device):
        from safetensors.torch import load_file

        state = load_file(str(MODEL_DIR / "model.safetensors"), device="cpu")
        self.device = device

        def get(name):
            return state[name].to(device=device, dtype=torch.float32).contiguous()

        p = "roberta.embeddings."
        self.word = get(p + "word_embeddings.weight")
        self.position = get(p + "position_embeddings.weight")
        self.token_type = get(p + "token_type_embeddings.weight")[0]
        self.emb_ln = (get(p + "LayerNorm.weight"), get(p + "LayerNorm.bias"))
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

    # -- weights in the canonical submission layout -----------------------
    def canonical(self, layer: int) -> torch.Tensor:
        """[4, d_model, d_model]: query, key, value, output.

        Rows index the model dimension, columns the head-major attention
        dimension (column = head * head_dim + offset).
        """
        params = self.layers[layer]
        return torch.stack([params["query"].t(), params["key"].t(), params["value"].t(), params["output"]])

    @staticmethod
    def to_module(matrix: str, canonical: torch.Tensor) -> torch.Tensor:
        return canonical if matrix == "output" else canonical.t()

    # -- forward ------------------------------------------------------------
    def embed(self, ids):
        not_pad = ids.ne(PAD_ID).int()
        positions = (torch.cumsum(not_pad, dim=1).type_as(not_pad) * not_pad).long() + PAD_ID
        hidden = self.word[ids] + self.position[positions] + self.token_type
        return F.layer_norm(hidden, (D_MODEL,), self.emb_ln[0], self.emb_ln[1], LN_EPS)

    def layer(self, hidden, keep, index, override=None):
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

    def mask_logits(self, hidden_at_mask):
        z = F.gelu(F.linear(hidden_at_mask, *self.head_dense))
        z = F.layer_norm(z, (D_MODEL,), self.head_ln[0], self.head_ln[1], LN_EPS)
        return F.linear(z, self.word, self.head_bias).float()

    def mask_log_probs(self, hidden_at_mask):
        return F.log_softmax(self.mask_logits(hidden_at_mask), dim=-1)

    def encode(self, batch, overrides=None, start_hidden=None, start_layer=0):
        """Encoder output for every position."""
        overrides = overrides or {}
        hidden = self.embed(batch["ids"]) if start_hidden is None else start_hidden
        for layer in range(start_layer, N_LAYERS):
            hidden = self.layer(hidden, batch["keep"], layer, overrides.get(layer))
        return hidden

    def run(self, batch, overrides=None, start_hidden=None, start_layer=0):
        """Hidden state at the mask position after the encoder."""
        hidden = self.encode(batch, overrides, start_hidden, start_layer)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, batch["mask_pos"]]

    def hidden_before(self, batch, layer):
        hidden = self.embed(batch["ids"])
        for index in range(layer):
            hidden = self.layer(hidden, batch["keep"], index)
        return hidden


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


def task_names() -> list[str]:
    return list(json.loads((TASK_DIR / "tasks.json").read_text())["tasks"])


def Task(name: str, device: torch.device, limit: int | None = None, token_budget: int | None = None):
    """The visible split of one task (multiple choice or open-ended generation)."""
    meta = json.loads((TASK_DIR / "tasks.json").read_text())["tasks"][name]
    if meta.get("type") == "generation":
        return GenerationTask(name, device, limit=limit, token_budget=token_budget)
    return ChoiceTask(name, device, limit=limit, token_budget=token_budget)


class ChoiceTask:
    def __init__(self, name: str, device: torch.device, limit: int | None = None, token_budget: int | None = None):
        from tokenizers import Tokenizer

        meta = json.loads((TASK_DIR / "tasks.json").read_text())["tasks"][name]
        rows = [json.loads(line) for line in (TASK_DIR / meta["file"]).open("r", encoding="utf-8")]
        if limit:
            rows = rows[:limit]
        tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        cands = []
        for word in meta["candidates"]:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            assert len(ids) == 1, word
            cands.append(ids[0])
        self.name = name
        self.rows = rows
        self.labels = [int(row["label"]) for row in rows]
        self.candidates = torch.tensor(cands, device=device)
        encoded = [tokenizer.encode(row["prompt"]).ids for row in rows]
        budget = token_budget or (32768 if device.type == "cuda" else 4096)
        order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), i))
        self.batches = []
        start = 0
        while start < len(order):
            size = max(1, budget // max(1, len(encoded[order[start]])))
            chunk = order[start : start + size]
            longest = max(len(encoded[i]) for i in chunk)
            ids = torch.full((len(chunk), longest), PAD_ID, dtype=torch.long)
            for pos, example in enumerate(chunk):
                ids[pos, : len(encoded[example])] = torch.tensor(encoded[example])
            self.batches.append(
                {
                    "index": chunk,
                    "ids": ids.to(device),
                    "keep": ids.ne(PAD_ID)[:, None, None, :].to(device),
                    "mask_pos": torch.tensor([encoded[i].index(MASK_ID) for i in chunk], device=device),
                    "labels": torch.tensor([self.labels[i] for i in chunk], device=device),
                }
            )
            start += size
        self._cache_layer = None
        self._cache = None

    def prefix(self, model: Roberta, layer: int):
        """Hidden states entering ``layer`` for every batch (cached)."""
        if self._cache_layer != layer:
            self._cache = None
            with torch.inference_mode():
                self._cache = [model.hidden_before(batch, layer) for batch in self.batches]
            self._cache_layer = layer
        return self._cache

    def evaluate(self, model: Roberta, overrides: dict | None = None, per_example: bool = False):
        """Accuracy (%), mean gold-token NLL (nats, clipped) on this split."""
        overrides = overrides or {}
        start = min(overrides) if overrides else N_LAYERS
        cache = self.prefix(model, start) if overrides else None
        n = len(self.labels)
        correct = [0] * n
        nll = [0.0] * n
        finite = True
        with torch.inference_mode():
            for b, batch in enumerate(self.batches):
                if overrides:
                    h = model.run(batch, overrides, start_hidden=cache[b], start_layer=start)
                else:
                    h = model.run(batch)
                cand = model.mask_log_probs(h).index_select(1, self.candidates)
                gold = cand.gather(1, batch["labels"][:, None])[:, 0]
                finite = finite and bool(torch.isfinite(cand).all())
                pred = cand.argmax(1).tolist()
                gold_nll = (-gold).double().tolist()
                labels = batch["labels"].tolist()
                for pos, example in enumerate(batch["index"]):
                    correct[example] = int(pred[pos] == labels[pos])
                    nll[example] = gold_nll[pos]
        clipped = [min(v, LOSS_CAP) if math.isfinite(v) else LOSS_CAP for v in nll]
        result = {"accuracy": 100.0 * sum(correct) / n, "loss": sum(clipped) / n, "finite": finite}
        if per_example:
            result["correct"] = correct
        return result


def laser_cleanup(text: str) -> str:
    """transformers 4.x ``clean_up_tokenization`` (applied by LASER's ``tokenizer.decode``)."""
    return (
        text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
        .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s")
        .replace(" 've", "'ve").replace(" 're", "'re")
    )


def _pack(encoded, reads, budget: int, device, golds=None) -> list[dict]:
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    batches = []
    start = 0
    while start < len(order):
        size = max(1, budget // max(1, len(encoded[order[start]])))
        chunk = order[start : start + size]
        longest = max(len(encoded[i]) for i in chunk)
        ids = torch.full((len(chunk), longest), PAD_ID, dtype=torch.long)
        rows, positions, gold = [], [], []
        for pos, example in enumerate(chunk):
            ids[pos, : len(encoded[example])] = torch.tensor(encoded[example])
            rows.extend([pos] * len(reads[example]))
            positions.extend(reads[example])
            if golds is not None:
                gold.extend(golds[example])
        batch = {
            "index": chunk,
            "ids": ids.to(device),
            "keep": ids.ne(PAD_ID)[:, None, None, :].to(device),
            "row": torch.tensor(rows, dtype=torch.long, device=device),
            "pos": torch.tensor(positions, dtype=torch.long, device=device),
            "count": torch.tensor([len(reads[i]) for i in chunk], dtype=torch.float64, device=device),
        }
        if golds is not None:
            batch["gold"] = torch.tensor(gold, dtype=torch.long, device=device)
        batches.append(batch)
        start += size
    return batches


class GenerationTask:
    """Open-ended answers generated at ``n_masks`` appended mask positions.

    Same definitions as the scorers: the argmax token at every mask is decoded
    and the example is correct when the (stripped, lower-cased) answer occurs
    in the (stripped, lower-cased) decoded text; the loss is the mean negative
    log-likelihood per answer token of "<question> <answer>" (no special
    tokens) with every answer token masked, clipped at 30 nats.
    """

    def __init__(self, name: str, device: torch.device, limit: int | None = None, token_budget: int | None = None):
        from tokenizers import Tokenizer

        meta = json.loads((TASK_DIR / "tasks.json").read_text())["tasks"][name]
        rows = [json.loads(line) for line in (TASK_DIR / meta["file"]).open("r", encoding="utf-8")]
        if limit:
            rows = rows[:limit]
        self.name = name
        self.rows = rows
        self.n_masks = int(meta["n_masks"])
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.answers = [row["answer"].strip().lower() for row in rows]
        self.labels = self.answers
        gen_encoded, gen_reads = [], []
        loss_encoded, loss_reads, loss_golds = [], [], []
        for row in rows:
            ids = self.tokenizer.encode(row["question"] + " " + " ".join(["<mask>"] * self.n_masks)).ids
            reads = [pos for pos, token in enumerate(ids) if token == MASK_ID]
            assert len(reads) == self.n_masks, row["id"]
            gen_encoded.append(ids)
            gen_reads.append(reads)
            claim = self.tokenizer.encode(row["question"] + " " + row["answer"], add_special_tokens=False).ids
            k = self._answer_len(claim, row["answer"])
            loss_golds.append(claim[len(claim) - k:])
            loss_encoded.append(claim[: len(claim) - k] + [MASK_ID] * k)
            loss_reads.append(list(range(len(claim) - k, len(claim))))
        budget = token_budget or (32768 if device.type == "cuda" else 4096)
        self.gen_batches = _pack(gen_encoded, gen_reads, budget, device)
        self.loss_batches = _pack(loss_encoded, loss_reads, budget, device, golds=loss_golds)
        self._texts: dict[tuple, str] = {}
        self._cache_layer = None
        self._cache = None

    def _answer_len(self, ids, answer: str) -> int:
        target = answer.strip()
        for start in range(len(ids) - 1, -1, -1):
            if self.tokenizer.decode(ids[start:], skip_special_tokens=False).strip() == target:
                return len(ids) - start
        raise AssertionError(f"answer {answer!r} not found")

    def _decode(self, tokens) -> str:
        tokens = tuple(tokens)
        if tokens not in self._texts:
            text = self.tokenizer.decode(list(tokens), skip_special_tokens=False)
            self._texts[tokens] = laser_cleanup(text).strip().lower()
        return self._texts[tokens]

    def prefix(self, model: Roberta, layer: int):
        """Hidden states entering ``layer`` for every generation and loss batch (cached)."""
        if self._cache_layer != layer:
            self._cache = None
            with torch.inference_mode():
                self._cache = (
                    [model.hidden_before(batch, layer) for batch in self.gen_batches],
                    [model.hidden_before(batch, layer) for batch in self.loss_batches],
                )
            self._cache_layer = layer
        return self._cache

    def evaluate(self, model: Roberta, overrides: dict | None = None, per_example: bool = False):
        """Accuracy (%), mean per-token answer NLL (nats, clipped) on this split."""
        overrides = overrides or {}
        start = min(overrides) if overrides else N_LAYERS
        gen_cache, loss_cache = self.prefix(model, start) if overrides else (None, None)
        n = len(self.rows)
        correct = [0] * n
        nll = [0.0] * n
        finite = True
        with torch.inference_mode():
            for b, batch in enumerate(self.gen_batches):
                if overrides:
                    hidden = model.encode(batch, overrides, start_hidden=gen_cache[b], start_layer=start)
                else:
                    hidden = model.encode(batch)
                logits = model.mask_logits(hidden[batch["row"], batch["pos"]])
                finite = finite and bool(torch.isfinite(logits).all())
                tokens = logits.argmax(dim=1).view(-1, self.n_masks).tolist()
                for pos, example in enumerate(batch["index"]):
                    correct[example] = int(self.answers[example] in self._decode(tokens[pos]))
            for b, batch in enumerate(self.loss_batches):
                if overrides:
                    hidden = model.encode(batch, overrides, start_hidden=loss_cache[b], start_layer=start)
                else:
                    hidden = model.encode(batch)
                log_probs = model.mask_log_probs(hidden[batch["row"], batch["pos"]])
                gold = log_probs.gather(1, batch["gold"][:, None])[:, 0].double()
                finite = finite and bool(torch.isfinite(gold).all())
                sums = torch.zeros(len(batch["index"]), dtype=torch.float64, device=gold.device)
                sums.index_add_(0, batch["row"], -gold)
                per_token = (sums / batch["count"]).tolist()
                for pos, example in enumerate(batch["index"]):
                    nll[example] = per_token[pos]
        clipped = [min(v, LOSS_CAP) if math.isfinite(v) else LOSS_CAP for v in nll]
        result = {"accuracy": 100.0 * sum(correct) / n, "loss": sum(clipped) / n, "finite": finite}
        if per_example:
            result["correct"] = correct
        return result


# --------------------------------------------------------------------------
# Submissions
# --------------------------------------------------------------------------


def write_patch(out_dir: Path, blocks: list[dict], notes: dict | None = None) -> None:
    """Write ``patch.json`` (structure only) for one experiment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"format": PATCH_FORMAT, "blocks": blocks}
    if notes:
        spec["notes"] = notes  # free-form; never read by the evaluator
    (out_dir / "patch.json").write_text(json.dumps(spec, indent=2) + "\n")


def load_json(path: Path, default):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
