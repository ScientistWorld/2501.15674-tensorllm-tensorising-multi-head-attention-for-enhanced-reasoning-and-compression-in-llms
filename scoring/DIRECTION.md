# Research Direction

## Research Problem

This benchmark studies post-training weight denoising and compression for transformer attention blocks. The scientific question is whether the internal structure of multi-head attention weights can be exploited to improve reasoning behavior without gradient training.

## Core Contribution

Improvements should focus on structured transformations of pretrained attention weights. The goal is to improve or refine how attention-layer weights are compressed, denoised, factorized, or reconstructed while keeping the model and task fixed.

## Approach Scope

In-bounds work may change the rank-selection strategy, tensor factorization algorithm, layer-selection policy, reconstruction objective, or structured parameterization of attention weights. It may also add method-side validation logic, as long as final scores are produced from the same artifacts and evaluation pipeline.

## Out of Bounds

Do not replace the task with supervised fine-tuning, prompt engineering alone, retrieval augmentation, ensembles, a larger model, or a new architecture. Those changes can improve FEVER accuracy but do not address the paper's research question about post-training structured attention-weight denoising and compression.
