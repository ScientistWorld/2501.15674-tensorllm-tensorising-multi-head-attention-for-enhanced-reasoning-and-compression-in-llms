# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-07 10:07] - method_runs
- Implemented TensorLLM for RoBERTa MHA layers: split Q/K/V/O weights by attention head, tensorised them into the paper's 4D layout, applied shared-factor Tucker/HOOI denoising over the non-head modes, and wrote reconstructed weights back to the model.
- Ran the method and original RoBERTa condition end-to-end on a balanced FEVER subset, producing prediction artifacts and evaluator scores without importing `method/` from `eval/`.

### [2026-06-07 10:20] - core_claim
- Broader FEVER/RoBERTa run on 320 balanced examples reproduced the paper's core claim at small scale: the TensorLLM-modified model improved accuracy over original RoBERTa while compressing the modified MHA layer.
- Validated the packaged gym with `python validate.py --compare`; scoring includes both benefit (`accuracy`, `loss`) and constraint (`mha_compression_ratio`) metrics.

## Stop Justification
- Completed at milestone `core_claim`.
