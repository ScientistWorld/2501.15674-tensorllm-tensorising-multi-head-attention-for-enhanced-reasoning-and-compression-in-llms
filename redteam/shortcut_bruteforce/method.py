"""Shortcut: exhaust both output candidates against public labels per case."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import AUDIT, ROOT, PROPOSED, read_rows, reference_artifacts, write_rows, write_json, evaluate
folder = AUDIT/'shortcut_bruteforce'
original, proposed, _ = reference_artifacts()
public = read_rows(ROOT/'data/fever_binary_dev.jsonl')
rows = []
trials = 0
for item in public:
    # Shortcut oracle is visible data, not a new reasoning method.
    best = None
    for answer in ['false', 'true']:
        trials += 1
        if answer == item['label']:
            best = answer
    rows.append({**item, 'prediction': best, 'nll': 0.0})
path = folder/'predictions.jsonl'
write_rows(path, rows)
proposed['predictions'] = str(path.relative_to(ROOT))
write_json(folder/'enumeration.json', {'candidates_per_case': 2, 'candidate_tests': trials,
                                     'oracle': 'public labels; shortcut, not model reasoning'})
evaluate(folder, {'conditions': [original, proposed]})
