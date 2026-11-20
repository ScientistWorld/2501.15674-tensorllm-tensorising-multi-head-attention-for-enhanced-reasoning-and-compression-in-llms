# Data

`tasks/` contains the tracked visible examples and their manifest. The three
visible splits contain 2,617 FEVER claims, 3,844 Bios biographies and 3,762
WikidataQA questions. Only these labelled examples may guide a submission.

`downloads/roberta-base/` contains the pinned public checkpoint fetched by
`scripts/download.sh`. All raw sources and preprocessing code capable of
regenerating held-out examples live under `eval/test/`. No method needs them.
