# Evaluation

## Metrics

Accuracy is the primary reasoning metric and is reported as the percentage of examples with the correct answer. For FEVER binary verification, a claim is predicted as true or false and compared against the gold label.

Loss is the mean negative log-likelihood of the correct answer under the two candidate labels. Lower loss indicates a more confident and better-calibrated prediction.

Compression ratio is the original parameter count of the modified model component divided by the size of its compressed representation. Higher is better, and values above 1 mean the reported modification uses fewer parameters than the original dense representation.

## Baselines and Targets

The main baseline is the original fixed pretrained masked language model with no post-training modification. The target is a small accuracy gain over the original model, lower loss, and a compression ratio above 1 for the modified component.

## Evaluation Protocol

The reusable benchmark evaluates a deterministic balanced subset of FEVER support/refute claims. Each claim is formatted as a binary verification prompt and scored by comparing the language model probabilities assigned to true and false at a masked answer position. `scripts/evaluate.sh` reads prediction artifacts produced by a candidate method and writes `scoring/scores.json`; it does not import or rerun the paper method.
