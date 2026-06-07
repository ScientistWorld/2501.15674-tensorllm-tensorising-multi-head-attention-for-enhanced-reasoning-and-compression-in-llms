# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-07 10:13] - method_runs
- Implemented the paper's multi-head tensorisation and shared-factor Tucker/HOOI reconstruction for RoBERTa MHA layers.
- Completed an offline FEVER smoke run that generated original-model and TensorLLM prediction artifacts.

### [2026-06-07 10:31] - core_claim
- Completed a broader 320-example FEVER/RoBERTa search over layers and Tucker ranks.
- Best TensorLLM condition improved accuracy from 50.00% to 54.69%, lowered loss from 1.097 to 0.696, and compressed the modified MHA layer by 1.99x.

### [2026-06-07 10:45] - core_claim
- Continuation audit confirmed the implementation is the paper's actual multi-head tensorisation plus shared Tucker/HOOI reconstruction, not a surrogate.
- Reran `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json` and `python validate.py --compare`; both passed with the same core-claim scores.

## Stop Justification
- Completed at milestone `core_claim`.
