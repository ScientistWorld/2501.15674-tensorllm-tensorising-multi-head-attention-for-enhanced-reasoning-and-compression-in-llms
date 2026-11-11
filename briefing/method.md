# Paper's Method

## Key Contribution

TensorLLM is a post-training denoising and compression method for the multi-head attention (MHA) weights of a transformer. It targets the attention block rather than the feed-forward block. It needs no extra data, training or fine-tuning.

## Approach

1. **Multi-head tensorisation.** Take one transformer layer. Split each of W^Q, W^K, W^V and (W^O)^T into its per-head sub-matrices (d_model x d_v each). Stack each head's four sub-matrices into a 3D tensor, then stack the heads into a 4D tensor of shape d_model x d_v x 4 x h.
2. **Tucker decomposition with shared factor matrices.** Approximate the 4D tensor as G x_1 U1 x_2 U2 x_3 U3, leaving the head mode uncompressed. All heads therefore share the factor matrices for the model dimension, the head dimension and the Q/K/V/O mode, and each head keeps its own core slice. The paper uses TensorLy's HOOI.
3. **Layer-selective application.** The reconstruction replaces the MHA weights of a single layer. The layer and the multilinear ranks are hyperparameters, chosen on a validation split (the first 20% of each dataset).

## Reproduction in this gym

- `method/tensorllm_roberta.py` covers every task:
  - it searches all 12 layers and 59 rank tuples, R1 in {64..512}, R2 in {16..64}, R3 in {2,3,4};
  - it runs HOOI with TensorLy's defaults;
  - it keeps the best visible-split accuracy, with lower loss breaking ties.
- It writes the selected structure as `outputs/submissions/tensorllm_roberta/<task>/patch.json`. The scorers run the same HOOI themselves.
- Selected structures:
  - FEVER: layer 0, ranks (128, 48, 3), compression 7.3x.
  - Bias in Bios: layer 7, ranks (64, 48, 4), compression 11.8x.

## Main Claims

The primary claim is that enforcing a shared higher-dimensional low-rank structure across the attention heads of a layer improves reasoning accuracy while compressing that layer's MHA weights. On RoBERTa the paper reports:

- FEVER: 50.0 → 50.45 accuracy, 3.74x compression;
- Bios: 64.5 → 72.57 accuracy, 8.78x compression.

Secondary claims:

- the method works on decoder-only models too (GPT-J, LLaMA2);
- it combines with FFN-side LASER;
- it beats compressing Q, K, V or O separately.
