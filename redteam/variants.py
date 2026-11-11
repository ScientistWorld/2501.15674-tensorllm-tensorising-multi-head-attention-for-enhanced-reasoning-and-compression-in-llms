"""Genuine method variants, all selection based only on visible even rows."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from harness import AUDIT, ROOT, BASE, PROPOSED, evaluate, read_rows, write_json, write_rows

spec = importlib.util.spec_from_file_location('audit_reference_method', ROOT/'method/tensorllm_roberta.py')
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)


def visible_summary(rows):
    visible = rows[::2]
    return {'accuracy': sum(r['prediction'] == r['label'] for r in visible)/len(visible),
            'loss': sum(r['nll'] for r in visible)/len(visible)}


def store_condition(folder, condition, score_key):
    rows = read_rows(ROOT/condition['predictions'])
    dest = folder/f'{score_key}.jsonl'
    write_rows(dest, rows)
    condition = copy.deepcopy(condition)
    condition['score_key'] = score_key
    condition['predictions'] = str(dest.relative_to(ROOT))
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    folder = AUDIT/args.name
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rows = method.load_records(str(ROOT/'data/fever_binary_dev.jsonl'), 320)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/downloads/roberta-base', local_files_only=True)
    base_model = AutoModelForMaskedLM.from_pretrained(ROOT/'data/downloads/roberta-base', local_files_only=True)
    cfg = SimpleNamespace(device='cpu', output_dir=str(folder/'inference'), batch_size=8, hooi_iters=5)
    if args.name == 'reference_seed0':
        original = method.run_condition(cfg, tokenizer, base_model, rows, BASE, None, None)
        original = store_condition(folder, original, BASE)
    else:
        from harness import reference_artifacts
        original = reference_artifacts()[0]
    if args.name == 'weaker_no_tensor':
        chosen = copy.deepcopy(original)
    elif args.name == 'weaker_rank_one':
        chosen = method.run_condition(cfg, tokenizer, base_model, rows, 'rank_one', 6, (1,1,1))
    elif args.name == 'better_more_hooi':
        cfg.hooi_iters = 20
        chosen = method.run_condition(cfg, tokenizer, base_model, rows, 'more_hooi', 6, (384,48,4))
    elif args.name == 'better_visible_search':
        cfg.hooi_iters = 10
        ranks = [(256,48,4), (320,48,4), (352,48,4), (384,48,4), (416,48,4), (448,48,4), (384,40,4), (320,56,4)]
        chosen = None
        best_value = None
        grid = []
        for layer in range(3,12):
            for rank in ranks:
                name = f'layer{layer}_r{rank[0]}-{rank[1]}-{rank[2]}'
                cond = method.run_condition(cfg, tokenizer, base_model, rows, name, layer, rank)
                summary = visible_summary(read_rows(cond['predictions']))
                entry = {'condition': cond, 'visible': summary}
                grid.append(entry)
                # Selection never accesses hidden scores or odd labels.
                value = (summary['accuracy'], -summary['loss'])
                if best_value is None or value > best_value:
                    best_value, chosen = value, cond
                write_json(folder/'search_progress.json', {'completed': len(grid), 'best': chosen, 'grid': grid})
                print(args.name, len(grid), name, summary, flush=True)
        write_json(folder/'selection.json', {'rule': 'max even-row accuracy; tie break even-row NLL',
                                           'trials': len(grid), 'chosen': chosen, 'best_visible': best_value})
    else:
        chosen = method.run_condition(cfg, tokenizer, base_model, rows, 'reference_layer6', 6, (384,48,4))
    proposed = store_condition(folder, chosen, PROPOSED)
    write_json(folder/'settings.json', {'seed': args.seed, 'hooi_iters': cfg.hooi_iters,
                                      'limit': len(rows), 'batch_size': 8, 'device': 'cpu',
                                      'torch_version': torch.__version__, 'selected': proposed})
    evaluate(folder, {'task': 'fever_binary_masked_lm', 'conditions': [original, proposed]})

if __name__ == '__main__':
    main()
