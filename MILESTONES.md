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

### [2026-06-07 11:01] - core_claim
- Repaired the official artifact evaluator so `scores.json` includes every declared metric: accuracy, balanced accuracy, loss, prediction-distribution shift, and MHA compression ratio.
- Local artifact-only validation passed; submitting a CPU test job to verify the packaged scripts inside the managed environment.

### [2026-06-07 11:05] - core_claim
- Final verification confirmed the method, evaluator, train/test slices, and download script are usable as a research gym.
- Cleaned method-agnostic documentation and scoring descriptions so from-scratch agents see the problem and constraints without implementation details.

### [2026-06-07 11:10] - core_claim
- Fixed the offline reproduction script so compute jobs do not attempt internet downloads.
- Added a short managed-container smoke job that regenerates method artifacts in an isolated output directory, then restores and validates the established core-claim scores.
- Repaired baseline-only scoring and added validation for invalid Tucker rank settings.

### [2026-06-07 11:13] - core_claim
- Final verifier checked folder separation, scoring files, train/test method-key consistency, artifact-only evaluation, and URL reachability.
- `python validate.py --compare` passes, and the reproduced TensorLLM score remains above the original RoBERTa baseline while reporting the compression constraint.

### [2026-11-11] - core_claim (hardening)
- Rebuilt the gym so the evaluator owns data, labels, decomposition, forward pass and parameter counts. Submissions declare attention-weight structure only.
- Reproduced TensorLLM at the paper's settings, selected on the visible split. Held-out results:
  - FEVER: 49.86 → 53.59, compression 7.3x;
  - Bias in Bios: 64.79 → 73.64, compression 11.8x (paper: 64.5 → 72.57, 8.78x).
- Demonstrated headroom: greedy multi-layer variant +2.55 ± 0.55 on Bios. Closed the redteam shortcuts plus label-trained core/factor shortcuts; see HARDENING.md.

### [2026-11-11] - core_claim (hardening round 2)
- Significance-based reasoning gate, with the worst loss on gate failure. Chance-level 64x compressors lost their credit (Bios 8/13 → 5/13 credited).
- Wider structure language (head subsets, axis-group layouts). The reference is reproduced exactly from the final code.

## Stop Justification
- Completed at milestone `core_claim`.
