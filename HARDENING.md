# Final hardening

This iteration addresses both `REDTEAM.md` and the follow-up `REVIEW.md`.
The final bundled validation is pending; its results will be added here after
the job. No performance gain below is claimed from an unrun patch.

## Changes decided before validation

The original hidden test accepted predictions and method-written losses and
compression. Earlier hardening replaced that path with fixed-model inference
and parameter accounting, but restricted submissions to evaluator-run numerical
approximations. The final interface also accepts a method's own data-free
program returning an explicit tensor network. The program receives an original
weight block, runs before any held-out examples are opened, and has no filesystem
access to workspace data. The evaluator independently loads its factors,
contracts them, counts every scalar, runs the fixed model and recomputes all
metrics. CPU execution is deterministic and has no wall-clock stopping rule.
Existing reference decompositions retain the paper's actual HOSVD-initialised
HOOI, 100 sweeps, tolerance 1e-8, with the head mode uncompressed.

Failed accuracy constraints now fail the entire run: every scored metric in
every experiment becomes accuracy 0, loss 30, compression 1. Missing experiments,
invalid representations, non-finite values and crashes follow the same rule.
A complete failure file is written before device/model startup, replacing stale
scores even if startup fails. Method discovery includes formerly ignored `_`
and `.` names. Missing keys from old visible score files are still evaluated
as failures. Unsafe path names, duplicate rank encodings and unknown patch/block
fields are rejected. Consequently forged numbers attached to a valid patch no
longer silently tie the reference.

All historical raw test predictions, answers, fitted-factor attack evidence and
regenerators were moved from `redteam/` into `eval/test/history/`; old executable
Python is retained as archival text. Generated evidence stays in
`eval/test/private/`. The public full-dataset generator was removed. Tracked
visible data ships with git, while public raw-data fetching and test regeneration
live exclusively in `eval/test/prepare_test.py`. Test output dumps are refused
outside `eval/test/`. `scripts/download.sh` fetches the pinned checkpoint with
real idempotent public commands; no committed code references shared storage.

The inherited WikidataQA addition directly addresses `REVIEW.md`: 3,762 visible
and 15,050 held-out factual statements, five-mask generation and answer-token
loss. Bios has 3,843 and FEVER 6,576 disjoint held-out examples. The final docs
now describe all three experiments. Bios/WikidataQA weights are 0.4 each; FEVER
is 0.2 because its paper gain is small and its observed headroom is weak.
Accuracy remains the main metric (coefficient 1), compression a headline
constraint (0.25), and loss secondary (0.1). The unchanged paper Table II values
are now under `_paper_table_ii` rather than pretending the enlarged gym is on
exactly the paper's evaluation scale.

## Before validation

Original `REDTEAM.md` reference on its tiny test: accuracy 56.875%, loss
0.6985086, compression 1.99478. Answer copying/enumeration, forged answers and
row dropping reached 100% and rank +3.7439; negative loss ranked +1.1471;
forged compression +0.7002; random balanced answers +0.2326. Omitting method
keys and ignored directories tied at 0. Empty/null predictions crashed without
complete worst values; NaN and stale scores could survive. The old direct-score
and success-flag attempts tied the reference because the obsolete path ignored
them. These are historical, differently scaled ranks, not comparable magnitudes
to the final three-task cohort.

Immediately before this iteration, the three-task reference measured:

| Experiment | Accuracy (%) | Loss | Compression |
|---|---:|---:|---:|
| Bios | 73.640385 | 5.303012 | 11.814438 |
| FEVER | 53.588808 | 1.147573 | 7.314014 |
| WikidataQA | 37.322259 | 15.245325 | 9.442018 |

The saved better multi-layer control improved Bios by 2.550 points (paired 95%
interval 1.48 to 3.62), with FEVER/WikidataQA unchanged. The other inherited
WikidataQA variant gained only 0.332 points (paired interval -0.032 to 0.696),
which does not establish an improvement. The bundled job adds visible-only
finer rank selection and a multilayer extension on the top three layers from
the original visible search, plus a custom core-rescaling procedure. No hidden
results influence their selection. All reference, baseline and control inference
will be rerun from the final evaluator, with the reference run alone first.

## Remaining limits

External public copies of these datasets, precomputed fitted parameters or
answers embedded in source code, and prohibited supervised training remain
research-scope violations for the separate attempt grader. Runtime dataset
access and ready-made fitted-factor artifacts are blocked, but source-level
semantic intent cannot be established by filesystem confinement. A numerically
equivalent custom decomposition is an honest tie, not a loophole. FEVER's weak
headroom remains a scientific limitation; its lower weight reflects that.
No defenses against leftover processes or altered Python environments were added.
