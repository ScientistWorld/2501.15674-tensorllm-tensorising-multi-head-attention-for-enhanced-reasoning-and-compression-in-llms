# Hardening report (iteration 11, rounds 1 and 2)

## Summary

**Round 1** rebuilt the gym so that its hidden test rewards scientific contribution. All of round 1's evidence still holds: the round-2 scorers materialise every round-1 patch bit-identically (`redteam/hardening/selftest.py`).

- **Submissions are structure only.**
  - A method no longer submits predictions, losses or parameter counts.
  - It submits a `patch.json` declaring:
    - which attention matrices of which layers to replace;
    - how to arrange them into a tensor;
    - a multilinear rank per compressed mode.
- **The evaluator does everything else.**
  - It runs HOOI with the paper's TensorLy defaults on the hash-pinned original roberta-base weights.
  - It writes the reconstruction into the model and runs the model on prompts that only `eval/test/` holds.
  - It computes accuracy, gold-token NLL and the compression ratio.
  - It owns the baseline, the reasoning gate, the 64x cap and the worst-value failure policy.
- **Data follows the paper's protocol.** FEVER and Bias in Bios go through LASER's preprocessing.
  - Visible split: the first 20% of each dataset.
  - Held-out splits: the FEVER paper_test half (6,576 claims) and the last 20% of Bias in Bios (3,843 biographies).
- **The reference is the paper's method at the paper's settings.** HOOI runs to convergence. Layer and ranks are searched on the visible split only. Bias in Bios (Table II's second RoBERTa dataset) was added.

**Round 2** fixed what round 1 still let through, or left inexpressible:

1. **Significance gate.** Loss and compression are credited only when the accuracy gain is statistically significant. Two one-sided tests must reach z >= 1.645:
   - a paired McNemar test against the unmodified model on the same examples;
   - a binomial test against the majority-class rate.

   Before, any gain counted, however small. On FEVER the unmodified model always answers the same class, and on Bios it sits near the majority rate. A chance-level structure could therefore clear the old gate by luck and collect up to 64x compression credit.
2. **Worst loss on gate failure.** A method that fails the gate now gets loss 30 instead of the unmodified model's loss. On Bios the paper's method raises loss (5.30 vs 4.88), so a gate-failing method used to beat the reference on loss.
3. **A wider structure language.** The decomposition stays evaluator-run.
   - Blocks may name a subset of `heads`.
   - `layout` may be any grouping of the four axes (target matrix, model dim, head, within-head dim) into tensor modes. `stack` and `heads` remain as shorthands.
   - Parameters are counted per (layer, matrix, head) slice.

   A method can now express head clustering, other tensorisations and partial-layer replacement, not only the reference's layer/rank settings.

All numbers below are held-out test unless noted. They come from `bash scripts/evaluate_test.sh` or from the same unchanged scorer with `--submissions outputs/hardening/submissions`.
- Round-2 raw outputs, patches and analysis: `redteam/hardening/results_round2/`. Pipeline: `redteam/hardening/run_round2.sh`; job c2c99342-c59, one A100, about 41 minutes.
- Round 1: `redteam/hardening/results/`.

**Rank deltas.** "Rank Δ" is the system formula: experiment weight × Σ_metrics |coefficient| × (improvement over the reference in the better direction) / (population std across the cohort). Real ranks depend on future cohorts.
- Unless noted, the cohort is the honest set: the unmodified model, the reference, the previous gym's config, low rank, LASER-MHA, both greedy variants, and the two new structural variants.
- `analysis_round2.json` also gives three other cohorts: with the 13 compressors, with failed runs (accuracy 0), and with both.

## Reference before and after

| | Old gym (before iteration 11) | After round 1 | After round 2 (final) |
|---|---|---|---|
| FEVER acc / loss / CR | 56.875 / 0.699 / 1.99 (160 odd rows, artifact scorer, selection used test rows) | 53.59 / 1.148 / 7.31 | **53.59 / 1.148 / 7.31** (unmodified 49.86 / 2.499) |
| Bios acc / loss / CR | not scored | 73.64 / 5.303 / 11.81 | **73.64 / 5.303 / 11.81** (unmodified 64.79 / 4.879) |
| Gate z (vs unmodified / vs majority) | – | – | FEVER 3.76 / 5.60; Bios 12.96 / 15.85 (visible: FEVER 2.32 / 3.25, Bios 14.15 / 15.29) |
| Paper, Table II | | | FEVER 50.0 → 50.45, loss 2.5 → 1.47, CR 3.74; Bios 64.5 → 72.57, loss 4.91 → 6.64, CR 8.78 |

**The reference re-run from the final code reproduces exactly.** The visible-split search is cached and the method code is unchanged. The new scorers return the same values, and the reference passes the stricter gate on both splits.

**Fidelity to the paper.**
- The unmodified model matches the paper almost exactly: FEVER 49.86 / 2.499 vs 50.0 / 2.5.
- The Bios gain matches the paper's size: +8.85 ± 0.67 here vs +8.07 in the paper.
- The FEVER gain (+3.73 ± 0.99) is larger than the paper's (+0.45).
- Bios loss rises with the method, as it does in the paper.

**Train vs test.** Visible and held-out agree: Bios 74.43 vs 73.64, FEVER 53.38 vs 53.59. The method is deterministic; seeds play no role.

## Do better results score higher? (variants, before → after)

"Before" is the round-1 scoring of the same raw outputs; materialisation is bit-identical, so this is exactly what the round-1 scorer writes. "After" is the round-2 scorer. Both use the gym's coefficients.

| Variant | Bios acc (paired Δ vs ref) | FEVER acc (paired Δ vs ref) | Rank Δ before → after |
|---|---|---|---|
| Reference (TensorLLM) | 73.64 | 53.59 | 0 → 0 |
| **Better: greedy multi-layer TensorLLM** (adds a layer only if visible accuracy rises ≥ 1 pt) | **76.19 (+2.55 ± 0.54)** | 53.59 (identical) | **+0.170 → +0.120** |
| Greedy, any visible gain (overfits) | 76.19 (+2.55 ± 0.54) | 53.48 (−0.11 ± 0.70); loss 4.55, CR 4.41 | −0.069 → −0.053 |
| New: head groups (reference ranks, heads in 2 data-free clusters, pre-declared) | 67.63 (−6.01 ± 0.61) | 51.35 (−2.24 ± 0.95) | −1.110 → −0.982 |
| New: Q/K and V/O tensorised separately (pre-declared) | 69.50 (−4.14 ± 0.51) | 49.86 (−3.73 ± 0.99, gate fails) | −1.476 → −1.535 |
| Weaker: key idea removed, LASER-style SVD of one matrix | 71.30 (−2.34 ± 0.55) | 49.82 (gate fails) | −1.598 → −1.572 |
| Weaker: reference layer at ranks (64,16,2) | 66.12 (−7.52 ± 0.65) | 49.83 (gate fails) | −1.319 → −1.401 |
| Unmodified model (baseline) | 64.79 (−8.85 ± 0.67) | 49.86 (−3.73 ± 0.99) | −1.795 → −1.788 |
| Previous gym's config (layer 6, (384,48,4)) | 42.49 | 47.11 | −3.737 → −3.857 |

**Where better methods rank.**
- The genuinely better method ranks first in every cohort, by more than the test's noise: Bios +2.55 ± 0.54, 95% CI 1.48 to 3.62.
- Every weaker method ranks below the reference, each by more than noise.
- The two new structural variants are honest ablations of the paper's design: joint tensorisation of all four matrices across all heads. They score 2 to 6 points below it, the direction of the paper's ablation (Table IV). So the wider language is scored sensibly.

## Shortcuts: fix and score before → after

### Round 2

**Chance-level heavy compression ("compress64").** Twelve one-layer Tucker structures at >= 64x, each choosing its ranks by visible accuracy, plus the best of them.

| | Before (round-1 gate) | After (significance gate) |
|---|---|---|
| Fix | | Loss and compression credited only for a significant gain |
| Bios compressors credited 64x | 8/13 | 5/13 (all with z ≥ 2.16) |
| FEVER compressors credited | 0/13 (all ≈ the constant baseline) | 0/13 |

The three compressors that lose their credit, `compress64_L0`, `L2` and `L3`, have Bios gains of z = 0.27, 0.14 and 0.41 over the unmodified model.

Their rank Δ before → after:
- with the compressors in the cohort: −2.27 / −2.19 / −2.10 → −2.61 / −2.61 / −2.60;
- with compressors and failed runs: −0.50 / −0.50 / −0.48 → −1.00 / −1.00 / −1.00.

Before, they outranked the unmodified model (−2.40 / −0.77) and tied or beat the honest LASER-MHA and low-rank variants (−2.09 / −2.19; −0.65 / −0.61), which have real, significant gains. After, they rank below all of them (−2.04 / −2.27 / −2.39; −0.77 / −0.72 / −0.78).

**Gate failure scored better than the reference on loss.**
- Fix: a gate failure now gets the worst loss (30).
- Before → after: `shortcut_collapse_compress` (rank-(1,1,1) Tucker of layer 11, raw CR 2,782) had a Bios loss of 4.879, better than the reference's 5.303; now 30.
- The same applies to every failed-gate row (previous config, LASER-MHA on FEVER, low rank on FEVER). Within an experiment, a gate-failing method is now dominated on every scored metric by any method that passes.

**Heavy compression with a real gain** (`shortcut_max_compression`: layer 9, ranks (16,16,4), raw CR 92 → capped at 64; Bios 67.86, z = 7.49).
- Not closed: it is a legitimate trade-off.
- Measured: rank Δ −1.90 → −1.98 (with compressors in the cohort), −0.42 → −0.52 (with failed runs too). It never outranks the reference or the better method.

**Malformed uses of the new fields:**
- Overlapping head slices (`shortcut_overlapping_heads`).
- A layout missing an axis (`shortcut_bad_axes`).
- Twelve per-head blocks that together exceed the dense size (`shortcut_slice_bloat`).

All three fail: worst values (0 / 30 / 1) in both experiments. The self-test also checks unknown, repeated, duplicate and boolean head or axis entries, and rank keys beyond the mode count.

### Round 1, re-scored with the final scorers (unchanged results)

| Shortcut (score before iteration 11) | Fix | After |
|---|---|---|
| Copy visible answers / enumerate labels against visible gold / forge gold labels (+3.74) | Methods cannot submit predictions or labels. Held-out labels exist only in hash-pinned `eval/test/data/`. | not expressible |
| Forged NLL, NaN or null loss (+1.15 / accepted / crash) | The evaluator computes the loss, clipped at 30; non-finite outputs mean failure. | not expressible |
| Forged compression metadata (+0.70) | Parameters are counted from the declared structure. | `shortcut_forged_metadata` = the reference exactly |
| Drop or repeat rows (+3.74) | Test set and size are fixed by the evaluator. | not expressible |
| Omit the method key (tie) / unlisted method (tie) | Every method directory on disk and every listed key is scored. Missing patch or experiment → worst. | worst (`shortcut_omitted_method`, `shortcut_empty`, `shortcut_missing_experiment` on Bios) |
| Replace or alter the fixed model (tie) | Hash check; on mismatch every method, the baseline included, scores worst. | all worst (`scores_test_tampered.json`, re-run in round 2) |
| Stale or pre-written score file (survived) | The scorer deletes its output first and always writes a full file. | overwritten (re-checked in round 2) |
| Chance-level labels plus invented compression (+0.23) | Distribution and balanced metrics removed, compression counted by the evaluator, gate. | closed; round 2 tightened the gate further |
| Label-trained cores or factor subspaces, found during round 1 (Bios +8.25 to +10.7, FEVER +10.4) | Structure-only submissions. `shortcut_trained_factors` is rejected. | worst |
| Non-compressing or impossible structures | Represented parameters must be fewer than dense; ranks are validated. | worst |

## Shortcuts left open, and why

1. **Choosing structures with visible labels.** This is the paper's own procedure, so it is in bounds. Over-selection on small visible gains is penalised by the held-out test (the any-gain greedy variant).
2. **Outside copies of the public datasets.** An agent with internet access could download FEVER and Bias in Bios and select structures against held-out labels. The leverage is bounded to a choice among discrete structures, and there is no continuous fitting. `CONSTRAINTS.md` forbids it, no file under `data/` holds held-out answers, and the separate grader checks.
3. **The decomposition itself is not method-supplied.** I did not reopen it.
   - Round 1 measured that label-trained cores and factor subspaces pay +8 to +11 points.
   - Such weights compress well: a few hundred parameters in a seeded random subspace can carry a label fit.
   - So neither a size cap on a submitted procedure or tensor, nor a bound on reconstruction error, nor file-access sandboxing would close this inside the evaluator.

   Instead, round 2 widened the structural language a method controls: head subsets, any axis grouping, rank on any mode, any number of blocks and layers. The selection procedure behind a patch is entirely the method's. Data-aware or non-Tucker decompositions remain inexpressible; this is a deliberate cost.
4. **A residual 5% chance per experiment** that a structure with no real gain passes the significance gate. This is the gate's nominal error rate.

## Headroom check

**Bounds.** No scored metric of the reference sits at its bound:

| Metric | Bios | FEVER | Bound |
|---|---|---|---|
| Accuracy | 73.6 | 53.6 | 100 |
| Loss | 5.30 | 1.15 | 0 |
| Compression | 11.8x | 7.3x | 64x cap |

**Measured headroom.** The greedy multi-layer variant beats the reference on Bios by +2.55 ± 0.54 and ranks first in every cohort. No simple variant reaches a bound.

**Brute force.**
- The brute force the paper avoids is training, which is not expressible.
- Enumerating structures on visible data is the paper's method; its held-out value is limited by generalisation.
- Pure compression maximisation (13 compressors) ranks far below the reference in every cohort.

**Baselines.** The unmodified model, single-matrix LASER and the low-rank variant all score clearly below the reference, on scales with room above it.

**Added experiment.** `bios_profession_roberta` was added in round 1. It is the paper's own dataset, the paper's gain there (+8 points) is far above test noise, and it is where the headroom lies.

## Weights and coefficients

**Experiments.** FEVER 0.5 and Bios 0.5: Table II reports both with equal standing.

**Metric coefficients.**
- Accuracy 1.0: the paper's "key evaluation metric".
- Compression 0.25: a headline claim of the paper.
- Loss 0.1: "included for completeness".

**Coefficients I checked but kept.** I tested compression at 0.1 against all cohorts. It changes no ordering among honest methods except one: the any-gain greedy variant rises from −0.053 to +0.039, still below the better method (+0.131). Compressors stay far below the reference under both values.
- The significance gate, not the coefficient, is what removes credit for compression without reasoning.
- So I kept 0.25, which reflects the paper's emphasis on compression.

## Not fixed / limitations, stated plainly

- **No FEVER headroom shown.** No variant beats the reference on FEVER:
  - greedy layers add nothing that survives the margin, and without the margin they add noise;
  - the new head-group and Q/K–V/O variants score lower;
  - all demonstrated headroom is on Bios.
- **The decomposition algorithm is fixed** (Frobenius-optimal HOOI), as explained above.
- **Scope.** Only RoBERTa is reproduced, with no GPT-J or LLaMA2, on 2 of the paper's 4 datasets. LASER-FFN hybrids (Table III) are out of scope because only attention matrices may change.
- **Prompt.** One trailing `<mask>`, as in LASER's released RoBERTa code; the paper's text mentions five masks. The unmodified-model numbers match the paper's to within 0.3 points, which suggests the paper used the same prompt.
- **Duplicate example.** One held-out Bios prompt duplicated a visible one and was dropped; it is listed in `eval/test/data/MANIFEST.json`.
- **Hardware.** GPU and CPU scoring can differ on a handful of borderline predictions. The hidden test takes under a minute on one A100 and roughly 30–60 minutes on a 1-CPU node.
