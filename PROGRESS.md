# Progress

## Current final gym (2026-11-20)

The final gym uses fixed-model inference on Bios, FEVER and WikidataQA, with
complete failure vectors and an open, confined data-free decomposition interface.
Bundled job b730d9d9-25d passed all integrity checks and reproduced the reference
exactly. A visible-selected Bios variant gains 2.55 points beyond paired noise.
The final accuracy-first coefficients give it rank +0.027936 (paired 95% interval
+0.015267 to +0.040605). Saved-output job 5471d878-b29 confirmed the final
rank, rejected all 17 archived raw-artifact submissions at admission, preserved
the reference-only official scores, and passed validation. The default job
entrypoint again reproduces the reference end to end.
WikidataQA-specific beyond-noise headroom requested in REVIEW remains unresolved.
See HARDENING.md for evidence and limits. The remaining entries below are historical.

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

Follow-up packaging audit on 2026-06-07 made `scripts/reproduce.sh` compute-node safe by removing runtime downloads and requiring `scripts/download.sh` to be run before submission. `scoring/TARGETS.md` now explicitly lists the balanced-accuracy and prediction-distribution robustness metrics that were already emitted by the evaluator.

The same audit made the evaluator score baseline-only manifests, added standard evaluation to `scripts/baseline.sh`, and added rank validation so impossible Tucker ranks cannot silently produce incorrect compression metadata.

## Deviations from Paper

The paper evaluates the full FEVER processed split and several large models. This reproduction starts with a deterministic balanced FEVER subset and RoBERTa-base to keep the first offline job lightweight. The algorithmic procedure is unchanged: post-training multi-head tensorisation with shared Tucker factors on an MHA layer.

The implementation reconstructs dense RoBERTa weights for ordinary Transformer inference after computing the shared Tucker representation. The reported MHA compression ratio therefore measures the factorized representation implied by the decomposition, not reduced in-memory parameters of the PyTorch module during inference.

## 2026-11-11 hardening (iteration 11)
- New evaluator-run structure-patch scorers (`eval/train`, `eval/test`).
- LASER-protocol FEVER and Bias in Bios splits; the held-out split is only under `eval/test/`.
- Paper-setting TensorLLM reference; variants and shortcuts audited through the hidden test.
- Full report in `HARDENING.md`.

## 2026-11-11 hardening round 2
- Reasoning gate is now significance-based (paired McNemar and binomial vs majority, z >= 1.645). A gate failure gets loss 30 and compression 1.
- Structure language widened: head subsets and any axis-group layout. The decomposition stays evaluator-run.
- Job c2c99342-c59:
  - self-test passed (round-1 patches bit-identical);
  - reference reproduced exactly (FEVER 53.59, Bios 73.64);
  - 13 compressors, 2 structural variants and 3 new shortcuts scored.
- Details in `HARDENING.md`.

## Final hardening (bundled job completed)

Read the paper, REDTEAM and REVIEW. Opened the data-free decomposition interface,
closed runtime label access and method-written metadata, enforced complete
all-task worst scores on failure, moved hidden-answer artifacts behind the test
boundary, and reconciled all three-task documentation and paper-number scaling.
The bundled job reran reference/baselines/controls, tested confinement and
malformed submissions, and measured paired headroom for visible-only refinements.
All checks passed; no new WikidataQA variant improved beyond paired noise.
The final evaluator-only weighting check reuses those saved outputs.
