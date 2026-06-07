# Progress

## What Works

The workspace now contains a TensorLLM implementation for RoBERTa attention layers. It tensorizes Q/K/V/O weights by head, applies shared-factor Tucker HOOI reconstruction over the non-head modes, writes the reconstructed weights back, and evaluates FEVER binary predictions from generated artifacts.

Audit on 2026-06-07 found that `eval/` remains separated from `method/`, `validate.py` passes, and the required RoBERTa/FEVER artifacts are present under `data/downloads/` for offline execution.

## Results

The first CPU smoke job completed successfully on 80 FEVER examples. The original RoBERTa condition reached 58.75% accuracy and 0.958 loss. The best TensorLLM condition preserved 58.75% accuracy, lowered loss to 0.840, and compressed the modified MHA layer by 3.19x. This establishes that the paper's method runs end-to-end and satisfies the compression/loss side of the claim, but it does not yet show an accuracy gain over the original model.

## Remaining

Run a broader layer/rank search and larger FEVER subset to look for the accuracy improvement needed for `core_claim`.

## Issues

The paper does not provide a code repository or exact rank/layer hyperparameters. The reproduction therefore implements the described algorithm directly and searches a small set of layer/rank settings.

The first evaluation wrote reconstruction error into `scores.json`, which is useful diagnostic metadata but not part of `reference.json`; this has been fixed so official scores only contain declared metrics.

## Deviations from Paper

The paper evaluates the full FEVER processed split and several large models. This reproduction starts with a deterministic balanced FEVER subset and RoBERTa-base to keep the first offline job lightweight. The algorithmic procedure is unchanged: post-training multi-head tensorisation with shared Tucker factors on an MHA layer.
