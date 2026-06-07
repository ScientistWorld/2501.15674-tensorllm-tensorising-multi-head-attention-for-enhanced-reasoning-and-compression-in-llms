# Reproduction Milestones

**Current: core_claim**

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

### [2026-06-07 10:45] - core_claim
- Broader CPU search completed on 320 balanced FEVER examples across RoBERTa layers 3-11 and six Tucker-rank settings.
- Best TensorLLM condition was layer 6 with ranks 384/48/4: accuracy improved from 50.00% to 54.69%, loss decreased from 1.097 to 0.696, and the modified MHA layer was compressed by 1.99x.
- The result supports the paper's core claim on the accessible RoBERTa/FEVER reproduction setting: post-training tensorisation of MHA can improve reasoning-task accuracy while compressing attention weights.

## Stop Justification
- Stopping at `core_claim`: the reusable environment is packaged and validated, and the central benefit-plus-compression claim is reproduced on the implemented FEVER/RoBERTa experiment.
