# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-07 10:05] - none
- Read the paper and identified TensorLLM's core algorithm: multi-head tensorisation of Q/K/V/O attention weights followed by shared-factor Tucker reconstruction.
- Implemented the initial RoBERTa/FEVER reproduction scaffold, but no compute job has completed yet.

### [2026-06-07 10:16] - none
- Audited the implementation and evaluation separation; the method is a direct implementation of the paper's tensorisation and shared Tucker reconstruction, and no reproduced scores exist yet.
- Narrowed the submitted CPU smoke job to a smaller layer/rank search so the first end-to-end run is more likely to complete on a test node.

### [2026-06-07 10:24] - method_runs
- CPU smoke job completed successfully and produced method artifacts plus `scoring/scores.json`.
- TensorLLM ran end-to-end on RoBERTa/FEVER, compressed the modified MHA layer by 3.19x, and lowered loss versus the original model, but did not improve accuracy on the 80-example smoke subset.

## Stop Justification
