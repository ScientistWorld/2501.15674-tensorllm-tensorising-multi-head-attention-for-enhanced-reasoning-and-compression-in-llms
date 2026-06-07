# Progress

## What Works

The workspace now contains a TensorLLM implementation for RoBERTa attention layers. It tensorizes Q/K/V/O weights by head, applies shared-factor Tucker HOOI reconstruction over the non-head modes, writes the reconstructed weights back, and evaluates FEVER binary predictions from generated artifacts.

Audit on 2026-06-07 found that `eval/` remains separated from `method/`, `validate.py` passes, and the required RoBERTa/FEVER artifacts are present under `data/downloads/` for offline execution. The evaluator reads prediction artifacts and independently checks the reported MHA compression ratio against the original and compressed parameter counts in the manifest.

## Results

The first CPU smoke job completed successfully on 80 FEVER examples. The original RoBERTa condition reached 58.75% accuracy and 0.958 loss. The best TensorLLM condition preserved 58.75% accuracy, lowered loss to 0.840, and compressed the modified MHA layer by 3.19x. This established `method_runs`.

The broader CPU search completed on 320 balanced FEVER examples. The original RoBERTa condition reached 50.00% accuracy and 1.097 loss. The best TensorLLM condition (`tensorllm_layer6_r384-48-4`) reached 54.69% accuracy and 0.696 loss while compressing the modified MHA layer by 1.99x. This supports `core_claim`: the paper's post-training tensorisation procedure improves the reasoning-task metric while preserving the compression constraint.

Continuation audit on 2026-06-07 reran the standalone evaluator and `validate.py --compare`; both passed and reproduced the same scores from existing method artifacts.

The main artifact evaluator now reports all declared metrics in `reference.json`: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio. A local artifact-only validation on 2026-06-07 11:01 UTC passed with TensorLLM at 54.69% accuracy, 54.69% balanced accuracy, 0.696 loss, 15.31 percentage-point prediction-distribution shift, and 1.99x MHA-layer compression versus RoBERTa at 50.00% accuracy, 50.00% balanced accuracy, 1.097 loss, 50.00 percentage-point shift, and 1.00x compression.

## Remaining

Higher milestones would require extending beyond the current single FEVER/RoBERTa setting: more of the FEVER split, additional model families, and reproduction of the paper's other reported tasks.

## Issues

The paper does not provide a code repository or exact rank/layer hyperparameters. The reproduction therefore implements the described algorithm directly and searches a small set of layer/rank settings.

The first evaluation wrote reconstruction error into `scores.json`, which is useful diagnostic metadata but not part of `reference.json`; this was fixed so official scores only contain declared metrics. The latest validated score file is produced by `scripts/evaluate.sh` from method artifacts, not copied from the paper.

The official evaluator initially omitted two declared constraint metrics from `scores.json`; this was repaired so the score contract now matches `reference.json`.

Final verification on 2026-06-07 cleaned method-agnostic documentation, corrected the loss metric coefficient direction, added notes for gym-only metrics, and reran artifact, train-slice, test-slice, and `validate.py --compare` checks successfully.

## Deviations from Paper

The paper evaluates the full FEVER processed split and several large models. This reproduction starts with a deterministic balanced FEVER subset and RoBERTa-base to keep the first offline job lightweight. The algorithmic procedure is unchanged: post-training multi-head tensorisation with shared Tucker factors on an MHA layer.
