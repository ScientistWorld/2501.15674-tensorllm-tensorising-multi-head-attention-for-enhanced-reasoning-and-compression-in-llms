Pre-run findings (no shortcut claimed successful without a run):

- Test scores artifact rows with odd indices, train scores even; neither binds IDs, claims, order, count or gold labels to a fixed dataset.
- Test trusts per-row nll verbatim; no probability consistency or finiteness checks. correct/accuracy flags in manifest are ignored.
- Compression consistency checks only divide two supplied numbers; fallback accepts compression_ratio alone without positive/finite checks.
- Test restricts method keys to mutable scoring/scores_train.json; fallback restricts to mutable scoring/scores.json. It does not discover all on-disk methods.
- Failures raise before writing output, with no worst-value fallback or atomic cleanup of stale score files.
- No reference metric is at its optimum: accuracy/balanced accuracy <100, loss/shift >0, compression has no declared upper bound.
- Reference all-row selection uses full 320 gold labels, subsequently split by parity. This contaminates the gym hidden half before audit.
- Algorithm inspected against local paper: correct shared Tucker non-head modes, Q/K/V/O tensorization, per-head cores, one-layer selection. Differences: 320 balanced first support/refute cases, no conflicting-claim filter, parity 50/50 test rather than last20%, one mask rather than five, binary normalized NLL rather than stated ground-truth token NLL, max192 tokens, limited layers/ranks/5 HOOI iterations, custom PyTorch rather than TensorLy, dense reconstruction rather than persisted compressed factors, only RoBERTa/FEVER.
- Real data and an observed directional effect, but not an independent or statistically convincing reproduction of paper magnitudes. 160 test cases yield .625 percentage-point resolution, exceeding paper RoBERTa FEVER gain .45 points.
- Baseline is original fixed RoBERTa near chance, no saturation. Full brute force over 160 binary labels is 2^160; the shortcut enumeration trial uses visible gold and demonstrates leakage rather than computational smallness.
