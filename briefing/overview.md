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
