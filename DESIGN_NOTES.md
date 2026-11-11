# Train/Test Evaluation Design Notes

Current design, from the iteration 11 hardening (details and evidence in `HARDENING.md`):

- **Submission.** `outputs/submissions/<method>/<experiment>/patch.json` declares structure only: layer, target attention matrices, layout (`stack` | `heads`), and ranks per compressed mode.
- **Scorers.** `eval/train/evaluate.py` (visible split, `data/tasks/`) and `eval/test/evaluate.py` (held-out split, `eval/test/data/`) are independent copies of the same logic. Each:
  - checks the pinned roberta-base hashes;
  - runs HOOI (TensorLy defaults) on the original weights for every declared block;
  - runs the model;
  - computes accuracy, gold-token NLL (full vocabulary) and the compression ratio.
- **Baseline and gate.** Both scorers own the unmodified-model baseline and the reasoning gate. Loss and compression count only when accuracy beats both the unmodified model and the majority-class rate. Compression is capped at 64x.
- **Failure.** Invalid or missing submissions get worst values (0 / 30 / 1).
- **Method discovery.** Every directory under `outputs/submissions/` is scored, plus every key listed in `scoring/scores.json` / `scoring/scores_train.json`.
- **Splits.** Visible: the first 20% of each LASER-ordered dataset. Held-out: the FEVER paper_test half (6,576 claims) and the last 20% of Bias in Bios (3,843 biographies).
- **History.** Earlier designs scored method-written prediction artifacts. Iteration 11 then tried two factor formats (v1 einsum factors, v2 projection factors) before settling on structure-only. Both factor formats were replaced because label-trained factors paid +8 to +11 points.
