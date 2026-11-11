# Evaluation

## What a method submits

A method is a directory `outputs/submissions/<method>/` with one sub-directory per experiment (`fever_roberta_binary/`, `bios_profession_roberta/`), each holding a `patch.json`:

```json
{
  "format": "mha-structure-patch-v3",
  "blocks": [
    {"layer": 2, "targets": ["value"], "layout": "stack", "ranks": {"0": 96}},
    {"layer": 10, "targets": ["query", "key"], "layout": "heads", "ranks": {"0": 256, "3": 6}},
    {"layer": 7, "targets": ["query", "key", "value", "output"], "heads": [0, 2, 4, 6, 8, 10],
     "layout": [["m"], ["t", "d"], ["h"]], "ranks": {"0": 64, "1": 96}}
  ],
  "notes": {"anything": "free-form, never read by the evaluator"}
}
```

A submission declares structure only. Each block names:

- `layer`: 0-11;
- `targets`: distinct names among `query`, `key`, `value`, `output`. Each matrix `W` is used in the orientation model-dim x attention-dim, with the attention dim ordered head by head (`query/key/value`: the transpose of the PyTorch `weight`; `output`: the PyTorch `weight` itself);
- `heads` (optional): a non-empty list of distinct head indices 0-11. Only those heads' columns of the target matrices are replaced. The default is all 12 heads.
- `layout`: how the block's weights are arranged into a tensor. They form a 4-axis array with axes
  - `t`: target matrix, in the order listed;
  - `m`: model dimension, 768;
  - `h`: head, the block's heads in increasing order;
  - `d`: within-head dimension, 64.

  A layout is an ordered list of tensor modes. Each mode is a list of axes that are merged into one mode (the first axis varies slowest). Every axis must be used exactly once. Two names are shorthands:
  - `"stack"` = `[["m"], ["h", "d"], ["t"]]`: a 768 x (64·heads) x n_targets tensor;
  - `"heads"` = `[["m"], ["d"], ["t"], ["h"]]`: a 768 x 64 x n_targets x heads tensor.
- `ranks`: a multilinear rank for each mode that is compressed (keys are mode indices). Modes without a rank are kept whole. Each rank must lie in [1, mode size] and must not exceed the product of the other modes' kept sizes.

The evaluator computes the representation itself. It runs a Tucker decomposition of the original pretrained tensor on the declared modes: HOOI starting from HOSVD, at most 100 sweeps, stopping when the relative error changes by less than 1e-8. It then writes the reconstruction back into the model. Each (layer, matrix, head) slice may be replaced by at most one block. A patch may hold up to 576 blocks, on any layers.

## Metrics (per experiment, on the held-out split)

- **accuracy** (primary): the percentage of examples whose gold answer token has the highest probability among the candidate tokens at the masked position.
- **loss**: the mean negative log-likelihood (nats) of the gold answer token under the full-vocabulary softmax at the masked position. It is clipped at 30 per example; lower is better.
- **mha_compression_ratio**: the dense parameter count of the 4 attention matrices of every modified layer, divided by the parameters of the submitted representation. For each block this is the factor matrices (mode size x rank per compressed mode) plus the core (the product of the kept sizes). Every head slice (768 x 64) of a modified layer that no block replaces counts as dense. The ratio is capped at 64.

**Reasoning gate.** Loss and compression only count when the accuracy gain is statistically significant. Two one-sided tests must both reach z >= 1.645 (5%) on the held-out examples:
- a paired McNemar test against the unmodified model on the same examples: z = (b - c) / sqrt(b + c), where b counts examples only the submission answers correctly and c those only the unmodified model answers correctly;
- a binomial test of the accuracy against the majority-class rate p0: z = (acc - p0) / sqrt(p0 (1 - p0) / n).

A submission that fails the gate keeps its accuracy, but its loss is set to the worst value (30) and its compression to 1.

**Failure.** A missing, unreadable or invalid patch scores accuracy 0, loss 30 and compression 1 in that experiment. So does a representation that is not smaller than the dense layers, non-finite outputs, an altered pretrained model or an evaluator crash.

## Baseline

`roberta_original`, the unmodified model, is always scored by the evaluator itself. A submission directory with that name is ignored.

## Protocol and splits

- Prompts follow LASER's preprocessing.
  - FEVER: "Consider the following claim: <claim>. Is this claim true or false. The claim is <mask>", with candidates ` true` and ` false`.
  - Bias in Bios: the last 50 tokens of a biography, followed by "The profession of the person in the text is called <mask>.", with 8 single-token profession candidates.
- Visible split (`data/tasks/`, labels included): the first 20% of each ordered dataset. That is 2,617 FEVER claims and 3,844 biographies.
- Held-out split, hidden: 6,576 FEVER claims and 3,843 biographies drawn from later parts of the same datasets.

## Commands

- `bash scripts/evaluate_train.sh` scores every method directory on disk, plus every method key listed in `scoring/scores.json` or `scoring/scores_train.json`, on the visible split. It writes `scoring/scores_train.json`.
- `bash scripts/evaluate.sh` does the same and writes `scoring/scores.json`.
- `bash scripts/evaluate_test.sh` runs the same scoring on the held-out split and writes `scoring/scores_test.json`. It takes no arguments.

The scorers verify the pretrained model files against pinned hashes. They need nothing from a method except `patch.json`.
