# Paper's Method

## Key Contribution

TensorLLM is a post-training denoising and compression method for transformer multi-head attention weights. It targets the attention block rather than the feed-forward block and requires no extra data, training, or fine-tuning.

## Approach

For one transformer layer, the method splits the query, key, value, and output projection matrices into per-head submatrices. For each attention head, the four submatrices are stacked into a 3D tensor. The per-head tensors are then stacked across heads into a 4D tensor.

The method applies Tucker decomposition to this 4D tensor while leaving the head mode uncompressed. All heads share the same factor matrices for the model dimension, head dimension, and Q/K/V/O matrix-type modes, while each head keeps its own core slice. The reconstructed low-rank tensor is written back into the original attention weights. The paper applies this layer-selectively and searches for layer/rank settings that improve validation performance.

## Main Claims

The primary claim is that enforcing a shared higher-dimensional low-rank structure across attention heads can improve reasoning accuracy while compressing the MHA weights of the modified layer.

Secondary claims are that the method works across encoder-only and decoder-only LLMs, can be combined with FFN-focused denoising methods such as LASER, and outperforms ablations that compress query, key, value, or output matrices separately.
