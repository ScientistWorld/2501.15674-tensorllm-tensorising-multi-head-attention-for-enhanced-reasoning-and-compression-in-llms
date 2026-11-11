"""Run an artifact method through the actual no-argument hidden script.

This file orchestrates tests; it does not implement or replace any evaluator.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'fever_roberta_binary'
PROPOSED = 'tensorllm_roberta'
BASE = 'roberta_original'
DEFAULT = ROOT / 'outputs/tensorllm_fever/manifest.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def condition(key, path, ratio=1.9947782091692483):
    return {'name': key, 'score_key': key, 'predictions': str(Path(path).relative_to(ROOT)),
            'original_mha_params': 2359296,
            'compressed_mha_params': 2359296 / ratio, 'compression_ratio': ratio}


@contextlib.contextmanager
def preserved():
    paths = [DEFAULT, *(ROOT/'scoring').glob('*.json')]
    saved = {p: p.read_bytes() if p.exists() else None for p in paths}
    # Scores are not in the scientist's starting view. Every run starts clean.
    for p in (ROOT/'scoring').glob('*.json'):
        p.unlink()
    try:
        yield
    finally:
        for p in (ROOT/'scoring').glob('*.json'):
            if p not in saved:
                p.unlink()
        for p, contents in saved.items():
            if contents is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(contents)


def shell_eval(script, folder, label):
    with (folder/f'{label}.stdout.txt').open('w') as out, (folder/f'{label}.stderr.txt').open('w') as err:
        result = subprocess.run(['bash', script], cwd=ROOT, stdout=out, stderr=err)
    source = ROOT/'scoring'/f'scores_{label}.json'
    if source.exists():
        shutil.copyfile(source, folder/f'scores_{label}.json')
    return result.returncode


def evaluate(folder, manifest, post_train=None, pre_test=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'manifest.json', manifest)
    with preserved():
        write_json(DEFAULT, manifest)
        if pre_test == 'direct_scores':
            write_json(ROOT/'scoring/scores_test.json', {'experiments': {EXPERIMENT: {'results': {
                PROPOSED: {'accuracy': 100, 'balanced_accuracy': 100, 'loss': 0,
                           'prediction_distribution_shift': 0, 'mha_compression_ratio': 1e15}}}}})
        train_rc = shell_eval('scripts/evaluate_train.sh', folder, 'train')
        if post_train:
            post_train()
        test_rc = shell_eval('scripts/evaluate_test.sh', folder, 'test')
        outcome = {'train_exit_code': train_rc, 'test_exit_code': test_rc,
                   'scores_test_exists': (ROOT/'scoring/scores_test.json').exists(),
                   'hidden_command': 'bash scripts/evaluate_test.sh',
                   'fresh_process': True}
        write_json(folder/'outcome.json', outcome)
    print(folder.name, outcome, flush=True)
    return outcome


def reference_artifacts():
    folder = AUDIT/'reference_seed0'
    manifest = read_json(folder/'manifest.json')
    original = next(c for c in manifest['conditions'] if c['score_key'] == BASE)
    proposed = next(c for c in manifest['conditions'] if c['score_key'] == PROPOSED)
    return original, proposed, read_rows(ROOT/proposed['predictions'])


def run_shortcut(name):
    folder = AUDIT/name
    folder.mkdir(parents=True, exist_ok=True)
    original, proposed, rows = reference_artifacts()
    rows = copy.deepcopy(rows)
    candidate = copy.deepcopy(proposed)
    path = folder/'predictions.jsonl'
    candidate['predictions'] = str(path.relative_to(ROOT))
    public = read_rows(ROOT/'data/fever_binary_dev.jsonl')
    post_train = None
    pre_test = None
    if name == 'shortcut_visible_answers':
        # Shortcut: the allegedly hidden gold labels ship in public data/.
        answers = {r['id']: r['label'] for r in public}
        for row in rows:
            row['prediction'] = answers[row['id']]
            row['nll'] = 0.0
    elif name == 'shortcut_forged_gold':
        # Shortcut: invent gold labels agreeing with a constant predictor.
        for row in rows:
            row.update(label='true', prediction='true', nll=0.0)
    elif name == 'shortcut_loss_metadata':
        # Shortcut: impossible negative NLL, unchanged model predictions.
        for row in rows:
            row['nll'] = -1000000.0
    elif name == 'shortcut_compression_metadata':
        # Shortcut: no artifact is checked against these self-reported counts.
        candidate.update(original_mha_params=2359296, compressed_mha_params=1,
                         compression_ratio=2359296.0)
    elif name == 'shortcut_drop_rows':
        # Shortcut: replace 320 records by four copies of easy public cases.
        chosen = []
        for label in ['false', 'true']:
            item = min((r for r in rows if r['label'] == label and r['prediction'] == label), key=lambda r:r['nll'])
            chosen.extend([copy.deepcopy(item), copy.deepcopy(item)])
        rows = chosen
    elif name == 'shortcut_omit_proposal':
        # Shortcut: keep a failed proposal on disk but remove its method key.
        for row in rows:
            row['prediction'] = 'false' if row['label'] == 'true' else 'true'
            row['nll'] = 30.0
        def post_train():
            p = ROOT/'scoring/scores_train.json'
            scores = read_json(p)
            scores['experiments'][EXPERIMENT]['results'].pop(PROPOSED)
            write_json(p, scores)
    elif name == 'shortcut_unlisted_method':
        # Shortcut: an extra on-disk method is never discovered by the test.
        def post_train():
            p = DEFAULT
            manifest = read_json(p)
            manifest['conditions'].append(condition('shortcut_unlisted_failed_method', path))
            write_json(p, manifest)
    elif name == 'shortcut_empty_predictions':
        rows = []
    elif name == 'shortcut_failure_stale':
        rows = []
        pre_test = 'direct_scores'
    elif name == 'shortcut_success_flags':
        for row in rows:
            row['correct'] = True
        candidate['accuracy'] = 1.0
        candidate['loss'] = 0.0
    elif name == 'shortcut_null_loss':
        for row in rows:
            row['nll'] = None
    elif name == 'shortcut_nan_loss':
        for row in rows:
            row['nll'] = float('nan')
    elif name == 'shortcut_direct_scores':
        pre_test = 'direct_scores'
    elif name == 'shortcut_constant':
        # Shortcut candidate with internally consistent, finite probabilities.
        for row in rows:
            row['prediction'] = 'true'
            row['nll'] = -math.log(0.51 if row['label'] == 'true' else 0.49)
        candidate.update(original_mha_params=2359296, compressed_mha_params=1,
                         compression_ratio=2359296.0)
    elif name == 'shortcut_balanced_constant':
        # Shortcut: input-independent balanced predictions and plausible NLL.
        for i, row in enumerate(rows):
            row['prediction'] = ['false', 'true'][(i//2) % 2]
            row['nll'] = math.log(2)
        candidate.update(original_mha_params=2359296, compressed_mha_params=1,
                         compression_ratio=2359296.0)
    elif name == 'shortcut_replace_fixed_model':
        pass
    else:
        raise ValueError(name)
    write_rows(path, rows)
    manifest = {'task': 'fever_binary_masked_lm', 'conditions': [original, candidate]}
    if name == 'shortcut_replace_fixed_model':
        # Shortcut: evaluator accepts cached outputs even after fixed model
        # configuration is replaced by an unusable one. Restore bytes finally.
        config = ROOT/'data/downloads/roberta-base/config.json'
        before = config.read_bytes()
        (folder/'original_config.json').write_bytes(before)
        try:
            write_json(config, {'model_type': 'shortcut_no_model', 'hidden_size': 1})
            write_json(folder/'replacement_config.json', read_json(config))
            return evaluate(folder, manifest)
        finally:
            config.write_bytes(before)
    return evaluate(folder, manifest, post_train, pre_test)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    args = p.parse_args()
    run_shortcut(args.name)

if __name__ == '__main__':
    main()
