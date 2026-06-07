# Evaluation

## Metrics

Accuracy is the primary reasoning metric and is reported as the percentage of examples with the correct answer. For FEVER binary verification, a claim is predicted as true or false and compared against the gold label.

Loss is the mean negative log-likelihood of the correct answer under the two candidate labels. Lower loss indicates a more confident and better-calibrated prediction.

MHA compression ratio is the original number of multi-head-attention parameters in the modified layer divided by the compressed representation size. Higher is better, and values above 1 mean the layer representation is smaller than the original dense weights.

## Baselines and Targets

The main baseline is the original pretrained RoBERTa model with no post-training modification. The target is the paper's RoBERTa FEVER result: a small accuracy gain over the original model, lower loss, and a multi-head-attention compression ratio above 1.

## Evaluation Protocol

The reusable benchmark evaluates a deterministic balanced subset of FEVER support/refute claims. Each claim is formatted as a binary verification prompt and scored by comparing the language model probabilities assigned to true and false at a masked answer position. `scripts/evaluate.sh` reads prediction artifacts produced by a candidate method and writes `scoring/scores.json`; it does not import or rerun the paper method.
