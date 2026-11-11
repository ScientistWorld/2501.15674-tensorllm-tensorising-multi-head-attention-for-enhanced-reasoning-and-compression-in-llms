"""Audit-side rank calculation. This never participates in hidden evaluation."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics
from harness import AUDIT, EXPERIMENT, PROPOSED, read_json, write_json

METRICS = read_json(AUDIT/'state/reference.json')['experiments'][EXPERIMENT]['metrics']
REFERENCE = read_json(AUDIT/'state/original_scores_test.json')['experiments'][EXPERIMENT]['results'][PROPOSED]
scenarios = {}
for folder in sorted(AUDIT.iterdir()):
    if not folder.is_dir() or not (folder/'outcome.json').exists():
        continue
    outcome = read_json(folder/'outcome.json')
    data = read_json(folder/'scores_test.json') if (folder/'scores_test.json').exists() else {}
    results = data.get('experiments', {}).get(EXPERIMENT, {}).get('results', {})
    own = results.get(PROPOSED, {})
    scenarios[folder.name] = {'outcome': outcome, 'metrics': own, 'all_method_keys': list(results),
                             'examples': data.get('metadata', {}).get('examples_per_method', {})}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def rank_cohort(names):
    scales = {}
    for metric in METRICS:
        values = [REFERENCE[metric]] + [scenarios[n]['metrics'][metric] for n in names
                                      if finite(scenarios[n]['metrics'].get(metric))]
        scales[metric] = statistics.pstdev(values) or 1.0
    ranks = {}
    for name, row in scenarios.items():
        contributions = {}
        for metric, definition in METRICS.items():
            value = row['metrics'].get(metric)
            if not finite(value):
                continue
            direction = 1 if definition['higher_is_better'] else -1
            contributions[metric] = abs(definition['coefficient'])*direction*(value-REFERENCE[metric])/scales[metric]
        nonfinite = [m for m in METRICS if m in row['metrics'] and not finite(row['metrics'][m])]
        valid = row['outcome']['test_exit_code'] == 0 and not nonfinite
        # A nonzero evaluator exit is not a successful scored attempt. The
        # ranking specification says to skip MISSING values, not NaN. Do not
        # invent either controller behavior in the audit's reported ranks.
        ranks[name] = {'delta_rank': sum(contributions.values()) if valid else None,
                       'contributions': contributions, 'rank_valid': valid,
                       'missing': [m for m in METRICS if m not in row['metrics']],
                       'nonfinite': nonfinite}
    return {'metric_scales': scales, 'ranks': ranks, 'cohort': names,
            'normalization': 'population standard deviation, weight=1, absolute coefficient; missing skipped; failed/nonfinite runs have no assigned rank'}

finite_names = [name for name, row in scenarios.items() if row['outcome']['test_exit_code'] == 0
                and all(finite(row['metrics'].get(m)) for m in METRICS)]
honest_names = [name for name in finite_names if not name.startswith('shortcut_')]
seed_names = [name for name in scenarios if name.startswith('reference_seed')]
spread = {}
for metric in METRICS:
    vals = [scenarios[n]['metrics'].get(metric) for n in seed_names]
    if vals and all(finite(v) for v in vals):
        spread[metric] = {'values': vals, 'min': min(vals), 'max': max(vals), 'range': max(vals)-min(vals)}
summary = {'reference_hidden_scores': REFERENCE, 'scenarios': scenarios, 'seed_spread': spread,
           'all_finite_cohort': rank_cohort(finite_names), 'honest_cohort': rank_cohort(honest_names),
           'caveat': 'Future ranked attempts and their metric standard deviations are unknown. These are audit-cohort scores; componentwise dominance is independent of positive finite scales.'}
write_json(AUDIT/'summary.json', summary)
lines = ['scenario\taccuracy\tbalanced_accuracy\tloss\tshift\tcompression\tlocal_rank_delta\ttest_exit']
ranks = summary['all_finite_cohort']['ranks']
for name, row in scenarios.items():
    vals = [row['metrics'].get(m, 'MISSING') for m in METRICS]
    lines.append('\t'.join(str(v) for v in [name, *vals, ranks[name]['delta_rank'], row['outcome']['test_exit_code']]))
(AUDIT/'summary.tsv').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines), flush=True)
