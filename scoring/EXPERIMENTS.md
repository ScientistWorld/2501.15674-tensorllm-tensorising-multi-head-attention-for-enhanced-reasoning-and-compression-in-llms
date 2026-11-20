# Experiment Overview

The experiments test whether a smaller representation of a fixed pretrained
model's attention projections improves its answers without training. Each
reports accuracy, gold-answer loss and compression. A result must improve
accuracy beyond paired uncertainty and the majority-answer rate on all tasks;
otherwise every scored metric receives its worst value on every task.

## bios_profession_roberta

Predict one of eight professions from a short biography. The skewed class
distribution makes a frequent-answer rule a strong simple baseline. The task
checks whether improvements use the biography rather than merely this prior.
There are 3,844 visible and 3,843 disjoint held-out biographies. Weight: 0.4.

## fever_roberta_binary

Decide whether a short factual claim is true or false. The original model is
near chance, so this task checks that an intervention recovers useful factual
signal and preserves it while addressing the other tasks. It has 2,617 visible
and 6,576 held-out claims. Weight: 0.2, reflecting its limited demonstrated
headroom and the paper's small effect on this task.

## wikidata_qa_roberta

Complete a factual statement with an open-ended answer. Five generated tokens
are checked for the answer; loss measures uncertainty per gold answer token.
This tests improvements beyond selecting between a small set of labels. It has
3,762 visible and 15,050 held-out statements. Its size supports paired detection
of sub-point changes and its original-model accuracy leaves substantial room
for progress. Weight: 0.4, equal to the other task with a large reported gain.

The evaluator-owned original model is the common baseline in every experiment.
Every ranked number comes from new inference with the supplied representation;
reported metrics, verdicts and selected subsets are never accepted.
