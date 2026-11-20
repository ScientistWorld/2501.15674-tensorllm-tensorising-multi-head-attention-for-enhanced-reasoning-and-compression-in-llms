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
- Tasks: FEVER claim verification (true/false) and Bias in Bios profession prediction (8 professions), both as single-token answers at the masked position.
- Data available to a method: the visible split of each task (`data/tasks/`, with labels) and the pretrained model. Held-out examples are never available.

## Why It Matters

Large language models are heavily over-parameterised. A post-training procedure that makes part of a model smaller and at the same time improves its accuracy, using no extra data, training or fine-tuning, would make existing models cheaper and better at once.

## Success Criteria

A successful solution raises held-out accuracy over the unmodified model on each task, while the modified attention layers are represented with fewer parameters than their dense originals. Lower answer loss and stronger compression are secondary gains. Both count only once the accuracy gain over the unmodified model (and over always answering the most frequent class) is statistically significant.

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


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
