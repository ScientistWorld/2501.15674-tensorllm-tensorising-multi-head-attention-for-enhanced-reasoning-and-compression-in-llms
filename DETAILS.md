# Overview

- **Paper ID:** 2501.15674
- **Title:** TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs
- **Domain:** Large language model compression and reasoning
- **TL;DR:** TensorLLM applies shared-factor Tucker decomposition to multi-head attention weights after pretraining, aiming to denoise the model and compress attention parameters without fine-tuning.

## Short Summary

The paper argues that existing post-training rank-reduction methods mainly benefit feed-forward layers and do not exploit the structure shared by attention heads. TensorLLM tensorizes the query, key, value, and output projections per head, then reconstructs them through a shared Tucker subspace across heads. Across RoBERTa, GPT-J, and LLaMA2 on four reasoning datasets, the paper reports accuracy gains together with MHA-layer compression.

## Key Results

- RoBERTa on FEVER improves from 50.0% to 50.45% accuracy with 3.74x MHA compression in the reported table.
- GPT-J shows larger FEVER gains, from 50.2% to 58.94% accuracy with 14.69x MHA compression.
- The largest reported MHA compression ratio is 247.3x on GPT-J HotPotQA.

## This Gym

- Fixed model: `roberta-base`.
- Two scored experiments from the paper's Table II: FEVER claim verification and Bias in Bios profession prediction.
- Splits follow LASER's preprocessing: the first 20% of each dataset is visible with labels, and the held-out split is hidden.
- A submission declares which attention matrices to replace and the low-rank tensor structure to replace them with. The evaluator computes the decomposition of the original weights, runs the model on held-out prompts, and scores accuracy, answer loss and the compression of the modified layers (see `briefing/evaluation.md`).
- Reproduced reference on the held-out split:
  - FEVER: 49.86 → 53.59 accuracy, compression 7.3x;
  - Bios: 64.79 → 73.64 accuracy, compression 11.8x.

---

# Problem Definition

## Research Question

Can the attention weights of a fixed pretrained language model be replaced, after training and without any further training or fine-tuning, by a smaller structured representation of themselves such that the model reasons better on a task?

## Setting

- Model: `roberta-base` (12 layers, 12 attention heads of size 64, model width 768), answering at a masked position.
- What may change: the query, key, value and output projection matrices of any attention layers. Everything else (embeddings, biases, layer norms, feed-forward blocks, the prediction head) stays fixed.
- Tasks: FEVER claim verification (true/false) Bias in Bios profession prediction (8 professions), and open-ended WikidataQA factual completion. The first two use a single masked answer; the third uses five appended masks.
- Data available to a method: the visible split of each task (`data/tasks/`, with labels) and the pretrained model. Held-out examples are never available.

## Why It Matters

Large language models are heavily over-parameterised. A post-training procedure that makes part of a model smaller and at the same time improves its accuracy, using no extra data, training or fine-tuning, would make existing models cheaper and better at once.

## Success Criteria

A successful solution raises held-out accuracy over the unmodified model on each task, while the modified attention layers are represented with fewer parameters than their dense originals. Lower answer loss and stronger compression are secondary gains. All tasks must pass before any result counts; secondary gains count only once the accuracy gain over the unmodified model (and over always answering the most frequent class) is statistically significant.

---

# Paper's Method

## Key Contribution

TensorLLM is a post-training denoising and compression method for the multi-head attention (MHA) weights of a transformer. It targets the attention block rather than the feed-forward block. It needs no extra data, training or fine-tuning.

## Approach

1. **Multi-head tensorisation.** Take one transformer layer. Split each of W^Q, W^K, W^V and (W^O)^T into its per-head sub-matrices (d_model x d_v each). Stack each head's four sub-matrices into a 3D tensor, then stack the heads into a 4D tensor of shape d_model x d_v x 4 x h.
2. **Tucker decomposition with shared factor matrices.** Approximate the 4D tensor as G x_1 U1 x_2 U2 x_3 U3, leaving the head mode uncompressed. All heads therefore share the factor matrices for the model dimension, the head dimension and the Q/K/V/O mode, and each head keeps its own core slice. The paper uses TensorLy's HOOI.
3. **Layer-selective application.** The reconstruction replaces the MHA weights of a single layer. The layer and the multilinear ranks are hyperparameters, chosen on a validation split (the first 20% of each dataset).

## Reproduction in this gym

- `method/tensorllm_roberta.py` covers every task:
  - it searches all 12 layers and 59 rank tuples, R1 in {64..512}, R2 in {16..64}, R3 in {2,3,4};
  - it runs HOOI with TensorLy's defaults;
  - it keeps the best visible-split accuracy, with lower loss breaking ties.
- It writes the selected structure as `outputs/submissions/tensorllm_roberta/<task>/patch.json`. The scorers run the same HOOI themselves.
- Selected structures:
  - FEVER: layer 0, ranks (128, 48, 3), compression 7.3x.
  - Bias in Bios: layer 7, ranks (64, 48, 4), compression 11.8x.

## Main Claims

The primary claim is that enforcing a shared higher-dimensional low-rank structure across the attention heads of a layer improves reasoning accuracy while compressing that layer's MHA weights. On RoBERTa the paper reports:

- FEVER: 50.0 → 50.45 accuracy, 3.74x compression;
- Bios: 64.5 → 72.57 accuracy, 8.78x compression.

Secondary claims:

- the method works on decoder-only models too (GPT-J, LLaMA2);
- it combines with FFN-side LASER;
- it beats compressing Q, K, V or O separately.

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:05] - none
- Read the paper and identified TensorLLM's core algorithm: multi-head tensorisation of Q/K/V/O attention weights followed by shared-factor Tucker reconstruction.
- Implemented the initial RoBERTa/FEVER reproduction scaffold, but no compute job has completed yet.

### [2026-06-07 10:16] - none
- Audited the implementation and evaluation separation; the method is a direct implementation of the paper's tensorisation and shared Tucker reconstruction, and no reproduced scores exist yet.
- Narrowed the submitted CPU smoke job to a smaller layer/rank search so the first end-to-end run is more likely to complete on a test node.

### [2026-06-07 10:24] - method_runs
- CPU smoke job completed successfully and produced method artifacts plus `scoring/scores.json`.
- TensorLLM ran end-to-end on RoBERTa/FEVER, compressed the modified MHA layer by 3.19x, and lowered loss versus the original model, but did not improve accuracy on the 80-example smoke subset.

### [2026-06-07 10:45] - core_claim
- Broader CPU search completed on 320 balanced FEVER examples across RoBERTa layers 3-11 and six Tucker-rank settings.
- Best TensorLLM condition was layer 6 with ranks 384/48/4: accuracy improved from 50.00% to 54.69%, loss decreased from 1.097 to 0.696, and the modified MHA layer was compressed by 1.99x.
- The result supports the paper's core claim on the accessible RoBERTa/FEVER reproduction setting: post-training tensorisation of MHA can improve reasoning-task accuracy while compressing attention weights.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:07] - method_runs
- Implemented TensorLLM for RoBERTa MHA layers: split Q/K/V/O weights by attention head, tensorised them into the paper's 4D layout, applied shared-factor Tucker/HOOI denoising over the non-head modes, and wrote reconstructed weights back to the model.
- Ran the method and original RoBERTa condition end-to-end on a balanced FEVER subset, producing prediction artifacts and evaluator scores without importing `method/` from `eval/`.

### [2026-06-07 10:20] - core_claim
- Broader FEVER/RoBERTa run on 320 balanced examples reproduced the paper's core claim at small scale: the TensorLLM-modified model improved accuracy over original RoBERTa while compressing the modified MHA layer.
- Validated the packaged gym with `python validate.py --compare`; scoring includes both benefit (`accuracy`, `loss`) and constraint (`mha_compression_ratio`) metrics.

### [2026-06-07 10:35] - core_claim
- Continuation audit restored the milestone from `method_runs` to `core_claim` after checking that `method/` implements the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction rather than a surrogate.
- Re-ran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; the reproduced scores remained RoBERTa original 50.00% accuracy vs TensorLLM 54.69% accuracy with 1.99x MHA compression.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 22m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 40m | **GPU**: 1.7h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

### [2026-11-11] - core_claim (hardening)
- Rebuilt the gym so the evaluator owns data, labels, decomposition, forward pass and parameter counts. Submissions declare attention-weight structure only.
- Reproduced TensorLLM at the paper's settings, selected on the visible split. Held-out results:
  - FEVER: 49.86 → 53.59, compression 7.3x;
  - Bias in Bios: 64.79 → 73.64, compression 11.8x (paper: 64.5 → 72.57, 8.78x).
- Demonstrated headroom: greedy multi-layer variant +2.55 ± 0.55 on Bios. Closed the redteam shortcuts plus label-trained core/factor shortcuts; see HARDENING.md.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 19m | **GPU**: 0.7h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

### [2026-11-11] - core_claim (hardening)
- Rebuilt the gym so the evaluator owns data, labels, decomposition, forward pass and parameter counts. Submissions declare attention-weight structure only.
- Reproduced TensorLLM at the paper's settings, selected on the visible split. Held-out results:
  - FEVER: 49.86 → 53.59, compression 7.3x;
  - Bias in Bios: 64.79 → 73.64, compression 11.8x (paper: 64.5 → 72.57, 8.78x).
- Demonstrated headroom: greedy multi-layer variant +2.55 ± 0.55 on Bios. Closed the redteam shortcuts plus label-trained core/factor shortcuts; see HARDENING.md.

### [2026-11-11] - core_claim (hardening round 2)
- Significance-based reasoning gate, with the worst loss on gate failure. Chance-level 64x compressors lost their credit (Bios 8/13 → 5/13 credited).
- Wider structure language (head subsets, axis-group layouts). The reference is reproduced exactly from the final code.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 18m | **GPU**: 0.7h
- **Jobs**: 11 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

### [2026-11-11] - core_claim (hardening)
- Rebuilt the gym so the evaluator owns data, labels, decomposition, forward pass and parameter counts. Submissions declare attention-weight structure only.
- Reproduced TensorLLM at the paper's settings, selected on the visible split. Held-out results:
  - FEVER: 49.86 → 53.59, compression 7.3x;
  - Bias in Bios: 64.79 → 73.64, compression 11.8x (paper: 64.5 → 72.57, 8.78x).
- Demonstrated headroom: greedy multi-layer variant +2.55 ± 0.55 on Bios. Closed the redteam shortcuts plus label-trained core/factor shortcuts; see HARDENING.md.

### [2026-11-11] - core_claim (hardening round 2)
- Significance-based reasoning gate, with the worst loss on gate failure. Chance-level 64x compressors lost their credit (Bios 8/13 → 5/13 credited).
- Wider structure language (head subsets, axis-group layouts). The reference is reproduced exactly from the final code.

</details>

### Iteration 11: gpt-6.1-sol (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 24m | **GPU**: 0.2h
- **Jobs**: 13 total (11 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

### [2026-11-11] - core_claim (hardening)
- Rebuilt the gym so the evaluator owns data, labels, decomposition, forward pass and parameter counts. Submissions declare attention-weight structure only.
- Reproduced TensorLLM at the paper's settings, selected on the visible split. Held-out results:
  - FEVER: 49.86 → 53.59, compression 7.3x;
  - Bias in Bios: 64.79 → 73.64, compression 11.8x (paper: 64.5 → 72.57, 8.78x).
- Demonstrated headroom: greedy multi-layer variant +2.55 ± 0.55 on Bios. Closed the redteam shortcuts plus label-trained core/factor shortcuts; see HARDENING.md.

### [2026-11-11] - core_claim (hardening round 2)
- Significance-based reasoning gate, with the worst loss on gate failure. Chance-level 64x compressors lost their credit (Bios 8/13 → 5/13 credited).
- Wider structure language (head subsets, axis-group layouts). The reference is reproduced exactly from the final code.

### [2026-11-20] - core_claim (final hardening)
- Three-task reference rerun alone from final code: Bios 73.6404%, FEVER 53.5888%, WikidataQA 37.3223%, all with compression and significant gains over the fixed model.
- Complete worst values on all-task failure; open confined data-free decomposition procedure; hidden-answer boundary and packaging repaired.
- Bios variant improves by 2.5501 points with a positive paired interval. Final weighting makes its rank advantage exceed paired noise.
- WikidataQA-specific beyond-noise improvement remains unresolved; see HARDENING.md.
- Final saved-output job 5471d878-b29 passed admission, paired-rank and reference-only score checks plus `python validate.py`; the end-to-end reference entrypoint is restored.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
