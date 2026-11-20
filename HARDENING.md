# Final hardening

The gym now rewards the demonstrated scientific improvement above the reference,
by more than paired test noise, and penalizes weaker and failed methods. The
reference still reproduces the paper's direction on all three tasks. The
WikidataQA-specific improvement requested by `REVIEW.md` is **not established**:
the new task is present and reproduced, but its best tested variant's gain is
within noise. This limitation is not counted as a closed loophole.

## Evidence and ranking

The bundled final-code job **b730d9d9-25d** succeeded. It ran the reference alone
first, then all controls, variants and fixed attacks together, without concurrent
training/search or a wall-clock stopping condition. Reference metrics were
identical between the exclusive and audit runs. Every method saw identical
inputs. Fixed weights, deterministic decomposition and frozen backbone remove
training-seed variation; paired intervals below measure finite-example noise.

The saved-output follow-up job **5471d878-b29** also succeeded. It recomputed
the final ranks and paired intervals without retraining or inference, rejected
all 17 archived raw-artifact submissions at admission, and verified that the
official train/test score files still contain only the reference and fixed
original-model comparator. `python validate.py` passed in both jobs.

[Final analysis](eval/test/final_results/analysis.json) contains the cohort,
paired accuracy and rank intervals, and before/after ranks on a common cohort.
[Before weighting](eval/test/final_results/analysis_before_weight_change.json)
preserves the first bundled audit. [All test results](eval/test/final_results/scores_test_all.json)
and [all visible results](eval/test/final_results/scores_train_all.json) retain
metric vectors and evaluator diagnostics. Raw per-example outputs stay only in
`eval/test/private/`. Official `scoring/scores_test.json`, `scores_train.json`
and `scores.json` contain only the independently rerun reference and the
fixed original-model comparator, never shortcut results.

The rank is the sum of experiment weight × absolute metric coefficient ×
better-direction improvement over the reference / population standard deviation
across the attempts ranked. Constant metrics contribute zero. It does not use
the sign of the stored coefficient to choose direction. Reported absolute ranks
depend on the named audit cohort; comparisons across different cohorts or the
old 160-example gym are not on a common numerical scale.

Experiment weights remain Bios 0.4, FEVER 0.2, WikidataQA 0.4. Accuracy's
coefficient remains 1 and loss 0.1. Compression's coefficient changes from 0.25
to **0.05** on every task. The paper calls accuracy the key evaluation metric;
loss is included for completeness. Compression remains both mandatory and
positively rewarded. In the initial audit, a 2.55-point accuracy improvement
and 1.06-nat loss improvement retaining 10.50× compression almost cancelled
against a reduction from 11.81× compression. Giving that marginal storage
tradeoff priority obscured a clear improvement in the paper's main quantity.
The final weight preserves the tradeoff while giving reasoning priority.
No task, answer, model, operating point or reference result changed for this
weighting adjustment; saved outputs were rescored without training or inference.

| Variant | Before final weighting: rank | Final rank | Final paired 95% rank interval |
|---|---:|---:|---:|
| Reference | 0 | 0 | [0, 0] |
| Better visible-selected multi-layer structure (`better_greedy_layers`) | +0.004078 | **+0.027936** | **[+0.015267, +0.040605]** |
| Finer visible rank search (`better_refined`) | +0.004078 | +0.027936 | [+0.015267, +0.040605] |
| Finer search with an extra WikidataQA layer | −0.214418 | −0.010608 | [−0.026004, +0.004788] |
| Lower-threshold greedy search on all tasks | −0.265426 | −0.015070 | [−0.033260, +0.003120] |
| Data-free core rescaling | −0.000702 | −0.000702 | [−0.008494, +0.007089] |
| Independently supplied equivalent decomposition | approximately 0 | approximately 0 | numerical roundoff only |
| Fixed unmodified model | −0.999818 | **−0.500573** | [−0.527295, −0.473850] |
| Severely weakened ranks | −3.054157 | **−2.554912** | failed run |
| Matrix-only LASER control | −3.054157 | **−2.554912** | failed run |

These before/after columns use the same 27-method final audit cohort. The finer
search selected no single-layer WikidataQA change meeting its predeclared
0.5-point visible-gain rule; it honestly reproduces the earlier better control.
All variant choices used visible data only. The rank standard error for the
better control is 0.006464; its final gain is 4.32 such standard errors. The
interval uses paired per-example accuracy/loss differences, including their
covariance. Compression is deterministic and contributes no sampling variance.

The better control's Bios accuracy is **76.190476%** versus **73.640385%**,
a paired gain of **2.550091 points**, SE 0.544347, 95% interval
**[1.483172, 3.617011]**. Loss improves from 5.303012 to 4.239876; compression
changes from 11.814438 to 10.495836. FEVER and WikidataQA are unchanged.
It remains far from an accuracy bound, so a simple variant has not exhausted
the test. The rescaling rule does not beat the reference. The original-model
and independent-matrix baselines remain clearly weaker.

## Reference before and after

Immediately before this iteration, the enlarged three-task reference had exactly
the following test values. The final exclusive run reproduces all of them
unchanged, including full precision in the score files; its rank remains 0.

| Experiment | Test size | Original accuracy | Reference accuracy | Original loss | Reference loss | Reference compression |
|---|---:|---:|---:|---:|---:|---:|
| Bios | 3,843 | 64.793130 | 73.640385 | 4.879054 | 5.303012 | 11.814438 |
| FEVER | 6,576 | 49.863139 | 53.588808 | 2.499153 | 1.147573 | 7.314014 |
| WikidataQA | 15,050 | 27.980066 | 37.322259 | 17.721584 | 15.245325 | 9.442018 |

The paper's direction holds: accuracy rises by 8.8473, 3.7257 and 9.3422 points
respectively, with parameter compression and no training. Bios loss rises, as
it does in the paper; this cost is retained. Visible reference accuracies are
74.427680, 53.381735 and 36.363636 respectively, comparable to held-out difficulty.
The all-four-matrix LASER baseline's raw test accuracies are 69.476971,
50.623479 and 34.445183, below the reference on every task. It fails the FEVER
constraint, so all its ranked metrics receive the complete failure vector.

For historical context, `REDTEAM.md` audited a different, contaminated tiny
FEVER test: reference accuracy 56.875%, loss 0.6985086, compression 1.99478.
Its prompting, loss and test size differ from this gym; those numbers were not
used as targets. `scoring/reference.json` now stores measured gym reference
values under `results` and keeps the paper's reported Table II numbers unchanged
under `_paper_table_ii` (and its different WikidataQA loss convention under
`_paper_loss_laser_convention`).

## Major shortcuts

The first table records original `REDTEAM.md` ranks, whose cohort and metric
set differ from the final audit. “Worst” means **accuracy 0, loss 30,
compression 1 in every one of the three experiments**, final audit rank
**−2.554912**. No missing entry, NaN or stale perfect score is admitted.
Historical raw artifacts were replayed against the hidden evaluator's admission
boundary; none is a valid patch. [Admission evidence](eval/test/final_results/legacy_admission.json)
distinguishes this parser replay from the full inference checks. Current-format
counterparts and the attacks below ran through the complete hidden test in the
bundled job.

| Major route | Before | Fix and after |
|---|---|---|
| Visible-answer copying and label enumeration | 100% accuracy; rank +3.7439 | Held-out answers and regenerators live only in `eval/test/`; no prediction interface. Historical artifacts rejected. Current custom code reading answers fails confinement: worst. |
| Forged gold labels and dropped/repeated rows | 100%; +3.7439 | Evaluator-owned hash-pinned files, row counts, prompts and labels. Submitted rows cannot select cases or change gold: worst. |
| Negative/null/NaN loss, false success flags | Negative loss ranked +1.1471; null crashed; NaN accepted; flags tied | Scores come from the fixed model's full-vocabulary probabilities. Unknown scored patch fields and non-finite networks fail. Legacy artifacts rejected; current forged metadata and NaN network: worst. |
| Invented compression and balanced random answers | Forged compression +0.7002; random answers +0.2326 | Count the actual representation, including dense untouched slices; require real accuracy gains on every task. Supplied counts/answers cannot change scores: worst. |
| Omit the method key or add an unlisted method | Tie at 0 | Discover every on-disk method directory, including `_`/`.` prefixes, and old visible score keys. Missing patch/experiment is a complete failure: worst. |
| Destroy the fixed model while keeping cached predictions | Tie at 0 | Hash-pin checkpoint/tokenizer; run model inference. Deliberate config tamper gave worst for every method and metric. |
| Empty outputs, stale/directly forged scores | Crash could preserve a perfect stale file; cached-output ties | Write complete worst values before device/model startup; replace scores atomically on success. Planted perfect omitted-method scores were overwritten with worst. Direct predictions are rejected. |

Additional weaknesses that still existed immediately before this final round
were measured on a common 20-method cohort. These before values use the old
compression coefficient; the final column uses the final 27-method cohort.

| Current-format route | Before rank | Fix | Final rank |
|---|---:|---|---:|
| Maximum compression despite failure on FEVER | **+0.275861** | Any failed task now fails all metrics on all tasks; no partial compression credit | **−2.554912** |
| Unknown forged fields attached to a valid reference patch | **0** | Strict patch/block schema rejects untrusted fields | **−2.554912** |
| Collapsed representation | −1.018406 | Complete failure vector replaces surviving accuracy/other-task metrics | −2.554912 |
| Missing experiment | −2.013741 | All-task failure and complete result coverage | −2.554912 |
| Invalid axes/layout, infeasible/duplicate ranks, overlapping heads, dense replacement, slice-count inflation, pre-fitted factor artifact | −2.597982 each | Validate structure, reject artifact factors, count dense slices and all stored scalars; all-task failure | −2.554912 each |
| Previously ignored directory | Ignored/tied | `_shortcut_unlisted` is discovered and failed | −2.554912 |
| New procedure reads held-out labels | Newly expressible | Denied workspace access; failure propagated to every metric | −2.554912 |
| New procedure returns a NaN tensor network | Newly expressible | Independently validate factors, contract and check finiteness | −2.554912 |

A task-failing submission now has the worst value of every scored metric on
all experiments. Since a valid solver improves those values, it cannot lose
to a failed method through compensation by another metric, regardless of
normalization or positive coefficient magnitudes.

## Interface and packaging changes

Earlier hardening already moved from trusted prediction artifacts to
fixed-model inference and restored the paper's method: multi-head tensorisation,
shared non-head Tucker factors with distinct head cores, HOSVD initialization
and HOOI (100 iterations, tolerance 1e-8). This round preserves that numerical
reference rather than substituting a different algorithm.

A submission may now provide its **own data-free procedure and tensor network**,
beyond choosing the reference's settings. A single Python program receives
only an original weight block and declared parameters, and returns an explicit
einsum plus factors. The evaluator independently loads finite tensors,
contracts them, verifies shape and counts every scalar. The program executes
before held-out examples are opened, in a CPU child whose filesystem access
is confined to system libraries and its scratch directory. Workspace and
answer access are denied; open dataset handles are not inherited. Seeds and
thread count are fixed. There is no wall-clock cutoff. Both train and test
passed reconstruction, denied-answer-read and NaN self-tests. An independently
implemented numerically equivalent procedure gives the same predictions and
scores up to roundoff, showing that the open interface is usable.

Every startup failure gets complete worst values. Unsafe method names,
noncanonical rank keys, unknown metadata and empty patches are rejected. All
old raw test predictions, answers, attack artifacts and answer regenerators
were moved from `redteam/` to `eval/test/history/`; obsolete Python is archival
text. Public full-dataset preparation code was removed. Test output dumps are
refused outside `eval/test/`; generated evidence is ignored under its private
subdirectory. Visible data is tracked and disjoint. The pinned model has real,
idempotent fetch commands in `scripts/download.sh`; test regeneration and raw
source fetching live under `eval/test/`. No live code references shared storage.

Train and test retain separate implementations, identical interfaces,
preprocessing and difficulty; neither imports `method/` or each other. The
briefing and all scoring documents now describe all three final experiments
without exposing the reference's algorithm. `scripts/evaluate_test.sh` and
`evaluate_train.sh` need no arguments and score all submitted directories.
`python validate.py` passed in both completed jobs. `scripts/run.sh` now invokes
`scripts/reproduce.sh` to reproduce the reference end to end; the audit and
saved-output rescore remain available as separate scripts.

## Headroom and remaining limits

No reference metric is at its favorable bound: every accuracy is below 100%,
every loss is above 0, and every compression ratio is below the cap of 64.
The Bios variant proves **ranked** headroom beyond paired noise, rather than
merely numerical room below a bound. The reference's gain over the fixed
unmodified model also exceeds noise on each task.

For `REVIEW.md`, the inherited WikidataQA addition is retained, rerun and fully
documented: 3,762 visible / 15,050 held-out statements, LASER preprocessing,
five-mask generation, and gold-answer-token loss. This reaches the paper's
open-answer setting beyond binary classification. FEVER's weight is reduced
to 0.2. Original-model and attention-only LASER baselines were rerun on it.
The finer single-layer grid, extra-layer search and new rescaling procedure
were all tried using visible selection. The best multi-layer variant gains
**0.332226 points**, paired SE 0.185558, interval **[−0.031467, +0.695919]**.
Thus **the requested beyond-noise improvement specifically on WikidataQA remains
unresolved**. Enlarging to all 15,050 unseen single-answer examples available
after the visible split did not resolve it; duplicating cases would not add
information. Its compression also drops to 2.125431, so this variant is not a
rank improvement. FEVER likewise has no demonstrated beyond-noise headroom.
The paper's Table III 37.40 involves combined feed-forward and attention
modifications; this gym remains an attention-only study. Other model families,
HotPotQA and actual compressed inference memory/latency are not reproduced.
Compression measures representation parameter count, not wall-clock speed.

Remaining prohibited shortcuts require the separate attempt grader: external
copies of public datasets, answers or fitted weights embedded in program text,
and supervised fitting disguised as a data-free algorithm. Runtime workspace
reads and ready-made fitted-factor artifacts are blocked, but program semantics
are not proven by filesystem confinement. A valid equivalent decomposition or
a visible-selected alternative with indistinguishable performance is an honest
tie, not a loophole. Label-free enumeration is not an independent factual oracle;
the paying exhaustive-label attack used leaked answers, and those inputs are
now inaccessible. No defenses against leftover processes or the Python
environment were added, as evaluation runs in a fresh container.
