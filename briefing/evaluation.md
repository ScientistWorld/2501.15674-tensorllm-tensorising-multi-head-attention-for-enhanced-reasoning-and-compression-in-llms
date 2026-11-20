# Evaluation

Run `bash scripts/evaluate_train.sh` for visible measurements and
`bash scripts/evaluate_test.sh` for held-out measurements. Both take no required
arguments, enumerate every directory in `outputs/submissions/` (including names
beginning with `_` or `.`), and include method keys from previous visible score
files. Results go under each experiment's `results` in `scoring/scores_train.json`
and `scoring/scores_test.json`. `scripts/evaluate.sh` writes the visible results
to `scoring/scores.json` for compatibility. Only these metric values are ranked.

A method supplies `outputs/submissions/<name>/<experiment>/patch.json` for all
three experiments: `bios_profession_roberta`, `fever_roberta_binary`, and
`wikidata_qa_roberta`. No prediction, answer, metric, success flag, checkpoint,
or method-supplied comparison model is accepted.

## Representation interface

```json
{"format":"mha-structure-patch-v3","blocks":[
  {"layer":4,"targets":["query","key","value","output"],
   "layout":"heads","procedure":"decompose.py","parameters":{"rank":32}}
]}
```

Each block names a layer (0–11), distinct projection names (`query`, `key`,
`value`, `output`), and optionally distinct `heads` (0–11; default all heads).
Canonical weights have axes `[t,m,h,d]`: target in listed order, model width
768, head in increasing order, and within-head width 64. Query/key/value use the
transpose of the PyTorch weight; output uses the PyTorch weight itself.

`layout` groups those four axes into modes, with each axis used exactly once.
`"heads"` means `[["m"],["d"],["t"],["h"]]`; `"stack"` means
`[["m"],["h","d"],["t"]]`. An explicit grouping such as
`[["m"],["t","d"],["h"]]` is also accepted. A slice can occur in only one block.

For an original weight tensor `x` in that layout, the supplied single source
file defines `decompose(x, parameters)`. It returns `(equation, tensors)`, where
`equation` is an explicit `torch.einsum` equation and `tensors` is a list of
floating-point tensors. For example, a matrix representation may return
`("ar,rb->ab", [left, right])`. The evaluator contracts the network and verifies
that the result has exactly the original shape and finite values. Every scalar
in every returned tensor counts as a stored parameter, including any scales or
constants. A factor cannot be declared free merely because it is shared or
sparse. This supports new numerical procedures and tensor networks, beyond the
built-in representation family.

Programs run on CPU with a fixed seed, original weights and their declared
parameters only. A filesystem sandbox denies workspace/dataset access. Only
system libraries and the private temporary directory are accessible. Programs
must be self-contained in one regular Python file (at most 1 MiB); parameters
must fit in 16 KiB. Use installed packages; no internet, training, fitted weight
arrays, extra files, or label access. There is no elapsed-time cutoff. Code
embedding fitted parameters or downloaded answers violates the research scope
and is also checked separately for each attempt.

For compatibility, a block can supply `ranks` instead of `procedure` and
`parameters`, e.g. `"ranks":{"0":96}`. This selects the fixed built-in numerical
approximation on those modes. Modes without ranks are retained. Ranks must be
canonical decimal mode keys and feasible positive integers. A custom procedure
can replace this computation entirely. Unknown patch/block fields fail; an
optional top-level `notes` object is unscored.

## Measurements and constraints

Every method is evaluated on identical fixed prompts with the same frozen
pretrained `roberta-base`. The evaluator verifies checkpoint hashes, rebuilds
all replacements, executes the model, and computes every scored number.

- **Accuracy (%)**, higher is better: FEVER selects false/true at one mask;
  Bios selects among eight professions at one mask. WikidataQA appends five
  masks, decodes their highest-probability tokens, and tests whether the gold
  answer occurs in that text, ignoring case.
- **Loss (nats)**, lower is better: full-vocabulary gold-token negative
  log-likelihood, clipped at 30 per example. WikidataQA masks every token of
  the answer appended to the question and averages token losses within each
  question, then over questions. It does not divide by an extra token.
- **MHA compression ratio**, higher is better: dense size of the four attention
  projections in every changed layer divided by the independently counted
  representation. Untouched head slices in changed layers count as dense.
  The representation must be strictly smaller. Credit is capped at 64.

Two accuracy constraints must pass on every task: a paired McNemar statistic
`(only_method_correct - only_baseline_correct) / sqrt(disagreements) >= 1.645`
against the evaluator-owned unmodified model, and a binomial statistic
`(accuracy - p0) / sqrt(p0*(1-p0)/n) >= 1.645` against the fixed majority rate.
For WikidataQA, `p0` is the frequency of the most common gold answer.

A failure of either constraint, invalid patch, missing experiment, non-finite
value, uncompressed representation, altered fixed model, or execution failure
fails the whole run. Every metric in every experiment then receives its worst
value: accuracy **0**, loss **30**, compression **1**. Raw diagnostics in metadata
are not ranked. The evaluator-owned `roberta_original` comparator reports its
actual measurements; submissions cannot replace it.

The rank sums experiment weight times the absolute metric coefficient times
improvement over the reference, with the sign determined by `higher_is_better`.
Each improvement is divided by that metric's population standard deviation
across the attempts being ranked. A constant metric contributes zero. Thus
coefficients set relative importance, independently of measurement units.
Accuracy has coefficient 1, compression 0.05, and loss 0.1. Experiment weights
are Bios 0.4, FEVER 0.2 and WikidataQA 0.4.

## Fixed splits

Visible examples and labels are tracked under `data/tasks/`: 3,844 Bios,
2,617 FEVER and 3,762 WikidataQA examples (the first 20% in source order).
Held-out files, their raw sources, regeneration code and per-example audit
outputs stay under `eval/test/`: 3,843 disjoint Bios, 6,576 FEVER and 15,050
WikidataQA examples. FEVER uses the full later paper-test half; WikidataQA uses
all examples after the visible first 20%, enlarging the paper's last-20% test.
The same preprocessing, prompts, numerical precision and definitions apply on
both sides. No model is trained during hidden evaluation. Decomposition runs
before held-out examples are opened and cannot access their files.
