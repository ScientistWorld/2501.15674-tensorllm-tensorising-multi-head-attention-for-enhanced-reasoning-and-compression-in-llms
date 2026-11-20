# Scientist Constraints

## Model

Use the fixed `roberta-base` checkpoint in `data/downloads/roberta-base` (its files are hash-checked by the scorers). Only the query, key, value and output projection matrices of the attention layers may change, and only through the submission format in `briefing/evaluation.md`. Switching models is resource scaling, not methodological progress.

## Data

Use only the visible splits in `data/tasks/` and the pretrained model. Do not fetch FEVER, Bias in Bios or WikidataQA from elsewhere, and do not use any other labelled data. The held-out examples come from the same public datasets, so using outside copies would amount to selecting on test labels.

## Training

The research question excludes training and fine-tuning: do not fit weights, factors or any other continuous parameters to labels. Using the visible split to choose among structures is allowed. That covers which layers, which matrices and heads, which arrangement and which ranks, as the paper does with its validation split. A submission may provide its own data-free numerical procedure. It receives only original weights in a confined process, and the evaluator independently reconstructs and counts its returned representation. Predeclared discrete hyperparameter selection on visible data is allowed; continuous label fitting and embedding fitted weights in source code are excluded.

## Evaluation

Report scores only through `scripts/evaluate_train.sh`, `scripts/evaluate.sh` and `scripts/evaluate_test.sh`. Do not edit `eval/`, the task data or the pretrained model files.

## Compliance Checklist

- [ ] Base model is the provided `roberta-base`, unmodified on disk
- [ ] Only `data/tasks/` (visible split) is used for any selection
- [ ] No gradient training or fitting of parameters to labels
- [ ] One `patch.json` per experiment under `outputs/submissions/<method>/`
- [ ] Scores come from the provided scorers
