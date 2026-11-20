# Research Direction

## Research Problem

This benchmark studies post-training structured denoising and compression of pretrained attention weights. The question is whether low-rank tensor structure that already exists in the weights of a fixed model can be exploited to improve its reasoning, without data beyond a small labelled validation split and without any training.

## Core Contribution

Progress means finding better structures in which to approximate the attention weights. That covers:
- which layers to modify, and how many;
- which projection matrices, and which heads, to group together;
- how to arrange them into tensors (any grouping of the matrix, model, head and within-head axes into tensor modes);
- which modes to compress, and to what ranks.

It also covers better principled or data-free rules for making those choices. A method may supply its own data-free decomposition procedure and tensor network, as well as its structural choices. The evaluator reconstructs the weights and counts every returned parameter. The built-in approximation remains available for compatibility.

## Approach Scope

In bounds:
- new arrangements of the attention matrices;
- groupings of heads that share a subspace;
- per-mode rank choices;
- new data-free numerical decompositions and explicit tensor networks;
- multi-layer combinations;
- selection strategies that use the visible split, or no labels at all;
- criteria that predict which layers or structures help.

## Out of Bounds

- Supervised fine-tuning or any gradient training against labels.
- External or held-out data.
- Prompt changes, a different or larger model, retrieval or ensembles.
- Changes to the feed-forward blocks or other non-attention parameters.

These can raise accuracy but do not address the research question.
