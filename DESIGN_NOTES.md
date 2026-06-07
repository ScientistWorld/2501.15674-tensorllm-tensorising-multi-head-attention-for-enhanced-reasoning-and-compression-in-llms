# Train/Test Evaluation Design Notes

## Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train wrapper calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`; the test wrapper calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked for hardcoded method names in the split wrappers. Both wrappers take a manifest path and the evaluators enumerate manifest conditions; they do not lock scoring to a single method default.
- Checked train/test leakage. The slice evaluators duplicate logic and do not import from each other. The test evaluator optionally reads `scoring/scores_train.json` only to keep the scored method keys aligned with the train slice.
- Checked that the split evaluators are attached to method outputs. They read prediction JSONL artifacts and compression metadata from the submitted manifest rather than regenerating predictions or importing `method/`.
- Checked for early score stubs. `scoring/scores_train.json` and `scoring/scores_test.json` were absent.
- Ran `bash -n` on both split wrapper scripts.

## Changed

- Added `balanced_accuracy` to both train and test evaluators. This makes one-class or majority-class shortcuts visible even when plain accuracy looks acceptable.
- Added `prediction_distribution_shift` to both train and test evaluators. This penalizes predicted label distributions that drift from the gold-label distribution on the scored slice, catching degenerate label-collapse shortcuts.
- Added required `description` fields to every nonzero metric in `scoring/reference.json`.
- Added the two new anti-shortcut metrics to `scoring/reference.json` and slightly increased the compression coefficient so the score still reflects the paper's accuracy-with-compression claim.

## Runtime Concerns For Runner Iteration

- The split is still a deterministic 50/50 row-index partition of the same prediction artifacts. This is lightweight and matched between train and test, but it is not an OOD split; it mainly detects methods that overfit the visible slice or optimize a degenerate metric.
- Prediction artifacts include labels and NLL values. The evaluator can check accuracy from labels and predictions, but it cannot independently verify that reported NLL values came from the model without rerunning inference.
- `balanced_accuracy` and `prediction_distribution_shift` assume the FEVER binary subset remains label-balanced enough for class-collapse detection to be meaningful. The runner should watch for saturation or tiny class counts in either slice.
- `scoring/reference.json` now contains split-harness metrics that the train/test evaluators emit. The legacy unsplit `scripts/evaluate.sh` path was intentionally left untouched in this design-only iteration.
