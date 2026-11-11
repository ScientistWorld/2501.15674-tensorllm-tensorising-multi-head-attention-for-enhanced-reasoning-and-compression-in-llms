# Problem Definition

## Research Question

Can the attention weights of a fixed pretrained language model be replaced, after training and without any further training or fine-tuning, by a smaller structured representation of themselves such that the model reasons better on a task?

## Setting

- Model: `roberta-base` (12 layers, 12 attention heads of size 64, model width 768), answering at a masked position.
- What may change: the query, key, value and output projection matrices of any attention layers. Everything else (embeddings, biases, layer norms, feed-forward blocks, the prediction head) stays fixed.
- Tasks: FEVER claim verification (true/false) and Bias in Bios profession prediction (8 professions), both as single-token answers at the masked position.
- Data available to a method: the visible split of each task (`data/tasks/`, with labels) and the pretrained model. Held-out examples are never available.

## Why It Matters

Large language models are heavily over-parameterised. A post-training procedure that makes part of a model smaller and at the same time improves its accuracy, using no extra data, training or fine-tuning, would make existing models cheaper and better at once.

## Success Criteria

A successful solution raises held-out accuracy over the unmodified model on each task, while the modified attention layers are represented with fewer parameters than their dense originals. Lower answer loss and stronger compression are secondary gains. Both count only once the accuracy gain over the unmodified model (and over always answering the most frequent class) is statistically significant.
