# Progress

## What Works

The workspace now contains a TensorLLM implementation for RoBERTa attention layers. It tensorizes Q/K/V/O weights by head, applies shared-factor Tucker HOOI reconstruction over the non-head modes, writes the reconstructed weights back, and evaluates FEVER binary predictions from generated artifacts.

Audit on 2026-06-07 found that `eval/` remains separated from `method/`, `validate.py` passes, and the required RoBERTa/FEVER artifacts are present under `data/downloads/` for offline execution.

## Results

No compute job has completed yet in this continuation. The next submitted job will run a CPU smoke search over a reduced set of layers/ranks and write `scoring/scores.json`.

## Remaining

Run the initial job, inspect whether TensorLLM improves over the original model on the FEVER subset, and expand the layer/rank search if the core claim is not yet supported.

## Issues

The paper does not provide a code repository or exact rank/layer hyperparameters. The reproduction therefore implements the described algorithm directly and searches a small set of layer/rank settings.

## Deviations from Paper

The paper evaluates the full FEVER processed split and several large models. This reproduction starts with a deterministic balanced FEVER subset and RoBERTa-base to keep the first offline job lightweight. The algorithmic procedure is unchanged: post-training multi-head tensorisation with shared Tucker factors on an MHA layer.
