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
