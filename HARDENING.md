# Hardening report (iteration 11)

## Summary

The gym was rebuilt so that its hidden test rewards scientific contribution.

- **Submissions are structure only.** A method no longer submits predictions, losses or parameter counts. It submits a `patch.json` naming which attention matrices of which layers to replace, how to arrange them into a tensor (`stack` or `heads`), and a multilinear rank per compressed mode.
- **The evaluator does everything else.**
  - It runs the decomposition itself on the hash-pinned original roberta-base weights: HOOI with the paper's TensorLy defaults (HOSVD start, at most 100 sweeps, tolerance 1e-8).
  - It writes the reconstruction into the model and runs the model on prompts only `eval/test/` holds.
  - It computes accuracy, gold-token NLL and the compression ratio.
  - It owns the baseline (the unmodified model), the reasoning gate, the 64x compression cap, and the worst-value failure policy.
- **Data now follows the paper's protocol.** FEVER and Bias in Bios use LASER's preprocessing, which the paper says it adopts. The visible split is the first 20% of each dataset, as in the paper. The held-out FEVER split is the paper_test half (6,576 claims, containing the paper's last 20%); held-out Bios is the paper's last 20% (3,843 biographies). Raw files that can regenerate test answers live only under `eval/test/`.
- **The reference is the paper's method at the paper's settings.**
  - HOOI runs to convergence; the old gym stopped after 5 sweeps.
  - Layer and ranks are searched (12 layers x 59 rank tuples) on the visible split only. The old reference was selected on rows later used as test.
- **Bias in Bios added.** It is Table II's second RoBERTa dataset, where the paper's effect is large (+8 points).

All numbers below come from `bash scripts/evaluate_test.sh`, or from the same unchanged scorer with `--submissions outputs/hardening/submissions`. Unless noted, every number is held-out test. Raw outputs, patches and logs are in `redteam/hardening/results/`. The pipeline is `redteam/hardening/run.sh` (variants in `redteam/hardening/variants.py`).

**Illustrative rank Δ.** "Rank Δ" is the system formula: each experiment's weight times the sum over its metrics of |coefficient| × (improvement over the reference in the better direction) / (population std across the honest cohort). The cohort is the unmodified model, the reference, the old reference config, the low-rank variant, the LASER variant, and both greedy variants. Real ranks depend on future cohorts. Failed shortcuts were not put in the cohort, since their 0/30 values would inflate σ.

## Reference before and after

| | Old gym (160 odd-row FEVER claims, artifact scorer) | Hardened gym, held-out |
|---|---|---|
| FEVER accuracy | 56.875 (unmodified 51.25) | **53.59** (unmodified 49.86) |
| FEVER loss / compression | 0.699 (binary-renormalised NLL) / 1.99x | **1.148** (paper's full-vocab NLL; unmodified 2.499) / **7.31x** |
| Bios accuracy / loss / compression | not scored | **73.64** / 5.303 / **11.81x** (unmodified 64.79 / 4.879 / 1) |
| Paper, Table II | FEVER 50.0 → 50.45, loss 2.5 → 1.47, 3.74x | Bios 64.5 → 72.57, loss 4.91 → 6.64, 8.78x |

**Fidelity to the paper.**
- The unmodified model matches the paper almost exactly: FEVER 49.86 / 2.499 vs 50.0 / 2.5.
- The Bios gain matches the paper's size: +8.85 here vs +8.07 in the paper. Bios loss rises with the method, as it does in the paper.
- The FEVER gain is larger than the paper's.

**Paired test noise.** These come from a bootstrap over held-out examples; every method sees the same prompts.
- Reference − unmodified: Bios +8.85 ± 0.67, FEVER +3.73 ± 0.99.
- On FEVER's paper-sized last 20% alone, the gap is +2.64 ± 1.50, which is not significant. This is why the FEVER test uses the whole paper_test half.
- The method is deterministic; seeds play no role.

**Train vs test.** Visible-split scores agree with held-out: Bios 74.43 vs 73.64, FEVER 53.38 vs 53.59.

## Do better results score higher? (variants, before → after)

The "before" rank Δ are the old gym's, from `REDTEAM.md`. In the old gym, no attempted improvement beat the reference.

| Variant | Before (old gym) | After: Bios acc / loss / CR | After: FEVER acc / loss / CR | Rank Δ after |
|---|---|---|---|---|
| Reference (TensorLLM) | 0 | 73.64 / 5.303 / 11.81 | 53.59 / 1.148 / 7.31 | 0 |
| **Better: greedy multi-layer TensorLLM** (adds a layer only if visible accuracy rises ≥ 1 pt) | n/a | **76.19** / 4.240 / 10.50 | 53.59 / 1.148 / 7.31 (no layer added) | **+0.151** |
| Greedy, any visible gain (overfits small gains) | n/a | 76.19 / 4.240 / 10.50 | 53.48 / 4.545 / 4.41 | −0.134 |
| Old "better" attempts (more HOOI sweeps; broader visible search) | −1.83, −2.71 | subsumed: HOOI now runs to convergence and the search covers the full grid | | |
| Weaker: key idea removed, LASER-style SVD of one matrix (best layer/matrix/rank on the visible split) | n/a | 71.30 / 6.536 / 1.32 | 49.82 / 4.495 / 1 (gate fails) | −1.511 |
| Weaker: reference layer at ranks (64,16,2) | −3.79 (ranks 1,1,1) | 66.12 / 5.210 / 31.56 | 49.83 / 2.499 / 1 (gate fails) | −1.227 |
| Weaker: old gym's reference config (layer 6, (384,48,4)) | 0 | 42.49 / 7.263 / 1 | 47.11 / 3.000 / 1 | −3.383 |
| Unmodified model (baseline) | −4.52 | 64.79 / 4.879 / 1 | 49.86 / 2.499 / 1 | −1.646 |

**Paired differences.**
- Greedy − reference on Bios: **+2.55 ± 0.55** (95% CI 1.48 to 3.64), with 440 discordant examples. It is identical to the reference on FEVER.
- LASER − reference: −2.34 ± 0.56 (Bios), −3.77 ± 0.99 (FEVER).
- Low rank − reference: −7.52 ± 0.65 (Bios), −3.76 ± 0.99 (FEVER).

So a genuinely better method scores higher, and the weaker ones score lower, each by more than the test's noise. The margin rule matters: accepting any visible-split gain added two FEVER layers (+0.8, +0.3 on visible). They did not transfer (−0.11 ± 0.68 on test), and the loss went up, so that variant ranks below the reference. The scorer separates the two honestly.

## Shortcuts: fix and score before → after

Rank Δ "before" is from `REDTEAM.md`, old gym. "After" is the hardened hidden test. "Worst" means accuracy 0, loss 30, compression 1 in that experiment; with this cohort it gives rank Δ −17.47 when both experiments fail.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Copy visible answers / enumerate labels against visible gold / forge gold labels | +3.74 each | Methods cannot submit predictions or labels. The evaluator runs the model on held-out prompts whose labels exist only in `eval/test/data/` (hash-pinned). The old 320-claim file with test answers was removed from `data/`. | not expressible |
| Forged NLL (−10⁶), NaN or null loss | +1.15 / accepted / crash | Loss is computed by the evaluator from model outputs, clipped at 30 per example; non-finite outputs mean failure. | not expressible |
| Forged compression metadata | +0.70 | Parameters are counted from the declared structure. `shortcut_forged_metadata` (reference structure plus forged accuracy, compression, "verdict" and submitted factor/core tensors) | **0.000** (identical to the reference) |
| Drop or repeat rows | +3.74 | The test set and its size are fixed by the evaluator (hash-pinned manifest). | not expressible |
| Omit the method key (inverted answers) | 0 (tie) | Every method directory on disk is scored, plus every key listed in `scoring/scores*.json`. A key without a patch, or a missing experiment, gets worst values. | **−17.47** (`shortcut_omitted_method`), **−4.67** (`shortcut_missing_experiment`, FEVER only) |
| Unlisted method | tie | Discovery from disk. | scored like any other method |
| Replace or alter the fixed model | tie | The model files are hash-checked; if they are altered, every method (baseline included) scores worst. | all **worst** (`scores_test_tampered.json`) |
| Stale or pre-written score file, failed run leaves forged scores | survived | The scorer deletes its output first and always writes a full file; a planted forged `scores_test_all.json` was overwritten. | overwritten |
| Empty predictions / empty directory | exit 1, no scores | Every failure path writes worst values. | **−17.47** (`shortcut_empty`) |
| Chance-level labels plus invented compression ("balanced constant") | +0.23 | Distribution-shift and balanced-accuracy metrics removed (not paper metrics, and gameable). Compression can't be invented. Compression and loss only count if accuracy beats both the unmodified model and the majority-class rate. Rank-(1,1,1) collapse of layer 11 (raw CR 2,782) | **−2.07** (gate fails on both, CR 1) |
| Constant prediction | −0.35 | same | covered by the collapse row |
| **New, found while hardening: core fine-tuned on visible labels** (v1 format accepted any core) | Bios **81.89** (+8.25 over reference) | v2/v3 formats carry no core: the evaluator computes it from the original weights. | not expressible |
| **New: factor subspaces trained on visible labels** (v2 format accepted factor matrices; the weights are nearly full-rank, so subspaces can rotate a lot at negligible reconstruction cost) | Bios **84.31**, FEVER **64.01** (+10.7 / +10.4; rank Δ +4.07) | v3 accepts structure only; the evaluator runs HOOI. `shortcut_trained_factors` (a factor-carrying patch) is rejected. | **−17.47** |
| **New: non-compressing or impossible structures** (full ranks, ranks exceeding the product of the other modes, rank 0, unknown layout) | n/a | Represented parameters must be fewer than the dense layers; ranks are validated. | **−17.47** each |
| **New: scorer infrastructure failure** (v1 wrote rebuilt weights to a 64 MB `/tmp`, failing later methods with "No space left on device") | n/a | Materialisation is in-process; no temp files. | fixed |

## Shortcuts left open, and why

1. **Choosing among structures with visible labels.** This is the paper's own procedure: layer and ranks are chosen on validation. More search over structures can overfit the visible split; the unmargined greedy variant shows the hidden test penalises it. It is in bounds.
2. **Outside copies of the public datasets.** FEVER and Bias in Bios are public. An agent with internet access could select structures against test labels. Mitigations:
   - no file under `data/` contains held-out answers;
   - submissions can only pick among discrete structures, so the leverage is bounded;
   - `CONSTRAINTS.md` forbids it, and the separate grader checks.
3. **Learning without labels on visible text** (e.g. picking structures by perplexity). This is legitimate.

## Headroom check

**Bounds.** No scored metric of the reference sits at its bound.

| | Bios | FEVER | Bound |
|---|---|---|---|
| Accuracy | 73.6 | 53.6 | 100 |
| Loss | 5.30 | 1.15 | 0 |
| Compression | 11.8x | 7.3x | 64x cap |

**Measured headroom.** A variant beats the reference by more than noise (greedy multi-layer, Bios +2.55 ± 0.55). No single simple variant reaches a bound.

**Brute force.**
- The brute force the paper avoids is training. It is not expressible.
- Enumerating structures on the visible split is the paper's method. Its test value is limited by generalisation: the unmargined greedy variant loses on FEVER.

**Baselines.** The unmodified model, the LASER-style single-matrix SVD and the old config all score clearly below the reference, on a 0-100 scale that leaves room above it.

**Added experiment.** `bios_profession_roberta` was added. It is the paper's own dataset where the paper's effect (+8 points) is far above test noise, and it is where the demonstrated headroom lies.

## Weights and coefficients (changed, and why)

- **Experiments.** FEVER 0.5, Bios 0.5. The paper reports both in Table II with equal standing. Both reproduce the paper's direction with gains above test noise.
- **Accuracy 1.0.** The paper calls accuracy "the key evaluation metric".
- **Loss 0.25 → 0.1.** The paper calls loss "an indirect performance measure … included for completeness", and its own method raises Bios loss.
- **Compression 0.25.** Compression is a claimed benefit, but it must not buy rank without reasoning. Hence the accuracy gate and the 64x cap.
- **Removed** `balanced_accuracy` and `prediction_distribution_shift`. They are not paper metrics, and the redteam showed the shift metric rewarded chance-level alternating labels.

## Not fixed / limitations, stated plainly

- **No FEVER headroom shown.** No tested variant beats the reference on FEVER. Greedy layers add nothing that survives the margin; without the margin they add noise. Demonstrated headroom is on Bios only.
- **The decomposition algorithm is fixed by the evaluator** (Frobenius-optimal HOOI). Future work can change structure but not the approximation objective; data-aware or non-Tucker decompositions are not expressible. This is the price of closing trained-factor shortcuts, which were indistinguishable from honest factors here.
- **Scope.** Only RoBERTa is reproduced (no GPT-J/LLaMA2), on 2 of the paper's 4 datasets. LASER-FFN hybrids are out of scope because only attention matrices may change.
- **Prompt.** One trailing `<mask>`, as in LASER's released RoBERTa code; the paper's text mentions five appended masks.
- **Duplicate example.** One held-out Bios prompt duplicated a visible one and was dropped (listed in `eval/test/data/MANIFEST.json`).
- **Hardware.** GPU and CPU scoring can differ on a handful of borderline predictions. On a 1-CPU node the hidden test takes roughly 30–60 minutes for the baseline plus one method; on one A100 it takes under a minute.
