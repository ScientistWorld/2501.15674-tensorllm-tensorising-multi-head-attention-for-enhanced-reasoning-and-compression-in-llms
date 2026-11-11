# Train/Test Evaluation Design Notes

Current design (iteration 11 hardening, rounds 1 and 2; details and evidence in `HARDENING.md`):

- **Submission.** `outputs/submissions/<method>/<experiment>/patch.json` declares structure only. Each block gives:
  - the layer and target attention matrices;
  - optionally a subset of heads;
  - a layout: `stack`, `heads`, or any grouping of the axes t/m/h/d into tensor modes;
  - ranks per compressed mode.
- **Scorers.** `eval/train/evaluate.py` (visible split, `data/tasks/`) and `eval/test/evaluate.py` (held-out split, `eval/test/data/`) are independent copies of the same logic. Each:
  - checks the pinned roberta-base hashes;
  - runs HOOI (TensorLy defaults) on the original weights for every declared block;
  - runs the model;
  - computes accuracy, gold-token NLL (full vocabulary) and the compression ratio, counted per (layer, matrix, head) slice.
- **Baseline and gate.** Both scorers own the unmodified-model baseline and the reasoning gate.
  - Loss and compression count only when the accuracy gain is significant: paired McNemar z >= 1.645 against the unmodified model, and binomial z >= 1.645 against the majority-class rate.
  - A gate failure keeps its accuracy, with loss 30 and compression 1.
  - Compression is capped at 64x.
- **Failure.** Invalid or missing submissions get worst values (0 / 30 / 1).
- **Method discovery.** Every directory under `outputs/submissions/` is scored, plus every key listed in `scoring/scores.json` / `scoring/scores_train.json`.
- **Splits.** Visible: the first 20% of each LASER-ordered dataset. Held-out: the FEVER paper_test half (6,576 claims) and the last 20% of Bias in Bios (3,843 biographies).
- **History.**
  - Earlier designs scored method-written prediction artifacts.
  - Round 1 tried two factor formats (v1 einsum factors, v2 projection factors) before settling on structure-only. Both factor formats were replaced because label-trained factors paid +8 to +11 points.
  - Round 2 replaced the strict "beats the unmodified model" gate with the significance gate. It also widened the structure language (head subsets, axis-group layouts); the decomposition itself stays evaluator-run.
