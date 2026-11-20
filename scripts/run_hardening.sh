#!/bin/bash
# One bundled validation job. No concurrent work and no wall-clock cutoff.
set -euo pipefail
cd /home/user
P=eval/test/private/final
mkdir -p "$P"
# Reference: paper procedure, fresh inference, fixed full visible selection.
# The numerical method/data are unchanged; cached visible search is reusable.
bash scripts/method.sh
python eval/train/evaluate.py --key-sources --dump-dir outputs/final_hardening/dumps_train_reference
cp scoring/scores_train.json "$P/scores_train_reference.json"
cp scoring/scores_train.json scoring/scores.json
python eval/test/evaluate.py --key-sources --dump-dir "$P/dumps_test_reference"
cp scoring/scores_test.json "$P/scores_test_reference.json"
# Apply every control, variant and fixed attack to the final evaluator together.
python eval/test/audit_final.py selftest
python scripts/hardening_variants.py
cp -r outputs/final_hardening/submissions/. outputs/submissions/
# Plant a stale perfect score and an omitted method key. A failed method must
# get complete worst values even when it exists only in a previous score file.
python - <<'PY'
import json
from pathlib import Path
p=Path('scoring/scores_train.json');d=json.loads(p.read_text())
for exp in d['experiments'].values():
 exp['results']['shortcut_omitted_method']={'accuracy':100,'loss':-1000000,'mha_compression_ratio':1e12}
p.write_text(json.dumps(d))
Path('scoring/scores_test.json').write_text(json.dumps(d))
PY
bash scripts/evaluate_test.sh --dump-dir "$P/dumps_test"
cp scoring/scores_test.json "$P/scores_test_all.json"
bash scripts/evaluate_train.sh --dump-dir outputs/final_hardening/dumps_train
cp scoring/scores_train.json "$P/scores_train_all.json"
python eval/test/audit_final.py tamper
python eval/test/audit_final.py analyze
# Leave only the independently rerun reference in ranked score files and on disk.
python - <<'PY'
from pathlib import Path
import shutil
for d in Path('outputs/submissions').iterdir():
 if d.name!='tensorllm_roberta':
  shutil.rmtree(d)
p=Path('eval/test/private/final')
for target,source in [('scores_test.json','scores_test_reference.json'),('scores_train.json','scores_train_reference.json'),('scores.json','scores_train_reference.json')]:
 shutil.copyfile(p/source,Path('scoring')/target)
PY
python validate.py
