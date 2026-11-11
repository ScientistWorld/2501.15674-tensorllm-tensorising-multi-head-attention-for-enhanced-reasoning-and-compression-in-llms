# Evaluation Targets

Every metric below is computed by the scorers from the submitted structure. They rebuild the weights, run the fixed model and count the parameters.

- Visible split: `bash scripts/evaluate_train.sh` (or `bash scripts/evaluate.sh`).
- Held-out split: `bash scripts/evaluate_test.sh`.

## Primary: fever_roberta_binary, bios_profession_roberta

Tests whether a post-training change of attention weights raises held-out task accuracy over the unmodified pretrained model.

- **Metric**: accuracy (higher is better)

## Constraint: fever_roberta_binary, bios_profession_roberta

Tests whether the modified attention layers are represented with fewer parameters than their dense originals. Submissions that are not smaller fail outright. Larger reductions are rewarded only together with an accuracy gain over the unmodified model and the majority-class rate, and are capped at 64x.

- **Metric**: mha_compression_ratio (higher is better)

## Secondary: fever_roberta_binary, bios_profession_roberta

Tracks the uncertainty of the gold answer on the same held-out examples. It is credited under the same accuracy gate.

- **Metric**: loss (lower is better)
