# Evaluation Targets

All three experiments (`bios_profession_roberta`, `fever_roberta_binary`,
`wikidata_qa_roberta`) measure:

- **accuracy**, higher is better: the fraction of correct held-out answers;
- **mha_compression_ratio**, higher is better: reduced attention storage,
  counted from the actual representation with dense untouched slices included;
- **loss**, lower is better: gold-answer uncertainty under the full vocabulary.

The task must show a significant paired accuracy gain over the fixed original
model and exceed the majority rate, while using fewer parameters. Failure gives
0 accuracy, 30 loss and 1 compression on every experiment. Compression credit
is capped at 64. Test sizes, comparators, thresholds and operating points are
fixed by evaluation. The three score files have exactly these targets, with
per-experiment `results` for each method.
