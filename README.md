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

---

# Problem Definition

## Research Question

Can a pretrained transformer language model be post-processed so that its reasoning-task accuracy improves while its attention-block parameters are compressed, without using additional training data or fine-tuning?

## Why It Matters

Large language models contain many redundant or noisy parameters. A good post-training procedure would make existing models smaller and, ideally, more accurate on reasoning tasks, without the cost and risk of retraining.

## Success Criteria

A successful solution should improve task accuracy over the original pretrained model on a fixed reasoning benchmark, avoid a large increase in prediction loss, and reduce the parameter count needed to represent the modified attention layer.

---

# Paper's Method

## Key Contribution

TensorLLM is a post-training denoising and compression method for transformer multi-head attention weights. It targets the attention block rather than the feed-forward block and requires no extra data, training, or fine-tuning.

## Approach

For one transformer layer, the method splits the query, key, value, and output projection matrices into per-head submatrices. For each attention head, the four submatrices are stacked into a 3D tensor. The per-head tensors are then stacked across heads into a 4D tensor.

The method applies Tucker decomposition to this 4D tensor while leaving the head mode uncompressed. All heads share the same factor matrices for the model dimension, head dimension, and Q/K/V/O matrix-type modes, while each head keeps its own core slice. The reconstructed low-rank tensor is written back into the original attention weights. The paper applies this layer-selectively and searches for layer/rank settings that improve validation performance.

## Main Claims

The primary claim is that enforcing a shared higher-dimensional low-rank structure across attention heads can improve reasoning accuracy while compressing the MHA weights of the modified layer.

Secondary claims are that the method works across encoder-only and decoder-only LLMs, can be combined with FFN-focused denoising methods such as LASER, and outperforms ablations that compress query, key, value, or output matrices separately.

---

# Evaluation

## Metrics

Accuracy is the primary reasoning metric and is reported as the percentage of examples with the correct answer. For FEVER binary verification, a claim is predicted as true or false and compared against the gold label.

Loss is the mean negative log-likelihood of the correct answer under the two candidate labels. Lower loss indicates a more confident and better-calibrated prediction.

MHA compression ratio is the original number of multi-head-attention parameters in the modified layer divided by the compressed representation size. Higher is better, and values above 1 mean the layer representation is smaller than the original dense weights.

## Baselines and Targets

The main baseline is the original pretrained RoBERTa model with no post-training modification. The target is the paper's RoBERTa FEVER result: a small accuracy gain over the original model, lower loss, and a multi-head-attention compression ratio above 1.

## Evaluation Protocol

The reusable benchmark evaluates a deterministic balanced subset of FEVER support/refute claims. Each claim is formatted as a binary verification prompt and scored by comparing the language model probabilities assigned to true and false at a masked answer position. `scripts/evaluate.sh` reads prediction artifacts produced by a candidate method and writes `scoring/scores.json`; it does not import or rerun the paper method.

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


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
