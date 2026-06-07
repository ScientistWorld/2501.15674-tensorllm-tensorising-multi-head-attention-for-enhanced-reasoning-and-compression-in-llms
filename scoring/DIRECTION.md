# Research Direction

## Research Problem

This benchmark studies post-training weight denoising and compression for pretrained language models. The scientific question is whether structure already present in model weights can be exploited to improve reasoning behavior without gradient training.

## Core Contribution

Improvements should focus on structured transformations of pretrained weights. The goal is to improve or refine how a fixed model component is compressed, denoised, or reconstructed while keeping the model and task fixed.

## Approach Scope

In-bounds work may change the component-selection policy, compression search strategy, reconstruction objective, or structured parameterization. It may also add method-side validation logic, as long as final scores are produced from the same artifacts and evaluation pipeline.

## Out of Bounds

Do not replace the task with supervised fine-tuning, prompt engineering alone, retrieval augmentation, ensembles, a larger model, or a new architecture. Those changes can improve FEVER accuracy but do not address the research question about post-training structured weight denoising and compression.
