# Evaluation Targets

## Primary: fever_roberta_binary

Tests whether a post-training modification improves binary fact-verification accuracy for a fixed pretrained language model.

- **Metric**: accuracy (higher is better)
- **Run**: `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json`

## Constraint: fever_roberta_binary

Tests whether the accuracy change is achieved while representing the modified attention layer with fewer parameters than the original dense MHA weights.

- **Metric**: mha_compression_ratio (higher is better)
- **Run**: `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json`

## Secondary: fever_roberta_binary

Tracks uncertainty on the same binary fact-verification examples.

- **Metric**: loss (lower is better)
- **Run**: `bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json`
