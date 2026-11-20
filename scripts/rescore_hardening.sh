#!/bin/bash
# Evaluation-only follow-up: reuse final saved raw inference, no retraining.
set -euo pipefail
cd /home/user
python eval/test/audit_final.py legacy-admission
python eval/test/audit_final.py analyze
python - <<'PY'
import json
from pathlib import Path
p=Path('eval/test/final_results')
a=json.loads((p/'analysis.json').read_text())
assert a['paired_rank_noise']['better_greedy_layers']['ci95'][0]>0
assert a['rank']['roberta_original']<0
assert a['all_failed_runs_have_complete_worst_values']
for split in ('train','test'):
    saved=json.loads((p/f'scores_{split}_reference.json').read_text())
    official=json.loads(Path(f'scoring/scores_{split}.json').read_text())
    assert official==saved
    for task,exp in saved['experiments'].items():
        assert set(exp['results'])=={'tensorllm_roberta','roberta_original'}
print('Final weighting exceeds paired noise; reference-only score files preserved')
PY
python validate.py
