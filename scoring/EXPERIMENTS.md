# Experiment Overview

Both experiments ask the same question on a different reasoning task: after the attention projections of a fixed pretrained `roberta-base` are replaced by a smaller structured representation of themselves, does the model answer held-out questions better than the unmodified model? Each experiment reports:
- held-out accuracy (the benefit);
- the answer loss (a secondary uncertainty measure);
- how many times smaller the modified attention layers became (the constraint, rewarded when it comes with a reasoning gain).

## fever_roberta_binary

Binary fact verification: the model decides whether a short factual claim is true or false at a masked answer position. The unmodified model is at chance here, so this experiment tests whether a weight-level change can unlock reasoning the model does not show out of the box. The held-out split has 6,576 claims, so a gain of about 2 accuracy points is distinguishable from noise.

## bios_profession_roberta

Profession prediction from a short biography, among eight professions, at a masked answer position. The class distribution is skewed (about 61% "professor"), and the unmodified model is only slightly above the majority rate. This experiment therefore tests whether a weight change makes the model use the text rather than its prior. The held-out split has 3,843 biographies.
