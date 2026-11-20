#!/usr/bin/env python3
"""Build final controls and visible-only refinement; no held-out data reads."""
from pathlib import Path
import json
import shutil
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.mlm_tasks import Roberta, Task, pick_device, load_json, save_json, MATRICES
from method.tensorllm_roberta import (decompose_layer, reconstruct_canonical, overrides_for,
                                     patch_for, compression_ratio, run_search, select, tensorize)
from method.network_tucker import decompose
OUT = ROOT / 'outputs/final_hardening'
SUBS = OUT / 'submissions'
REFERENCE = ROOT / 'outputs/submissions/tensorllm_roberta'
CONTROLS = ROOT / 'baseline/controls'


def write(name, task, patch):
    d = SUBS / name / task
    d.mkdir(parents=True, exist_ok=True)
    (d/'patch.json').write_text(json.dumps(patch, indent=2)+'\n')
    return d


def main():
    SUBS.mkdir(parents=True,exist_ok=True)
    for d in CONTROLS.iterdir():
        shutil.copytree(d, SUBS/d.name, dirs_exist_ok=True)
    shutil.copytree(REFERENCE,SUBS/'tensorllm_roberta',dirs_exist_ok=True)
    device=pick_device()
    model=Roberta(device)
    name='wikidata_qa_roberta'
    task=Task(name,device)
    # Original method, finer ranks, same full visible split and solver. Layers
    # are the top three distinct layers by the original *visible* search.
    log=load_json(ROOT/'outputs/tensorllm_search/search_log.json', {'conditions':{}})
    entries=sorted((e for e in log['conditions'].values() if name in e),
                   key=lambda e:(e[name]['accuracy'],-e[name]['loss']),reverse=True)
    layers=list(dict.fromkeys(e['layer'] for e in entries))[:3]
    grid=[(r1,r2,r3) for r1 in (16,24,32,48,56,72,80,96) for r2 in (32,48,64) for r3 in (3,4)]
    refined=run_search(model,{name:task},layers,grid,OUT/'refinement.json',100,1e-8)
    best=select(refined,name)
    refsel=load_json(REFERENCE/'selection.json',{})
    # A predeclared selection rule; the hidden test is not involved.
    chosen=best if best[name]['accuracy'] >= refsel[name]['validation']['accuracy']+0.5 else None
    shutil.copytree(SUBS/'better_greedy_layers',SUBS/'better_refined',dirs_exist_ok=True)
    if chosen:
        write('better_refined',name,{'format':'mha-structure-patch-v3',
            'blocks':[patch_for(chosen['layer'],chosen['ranks'])],
            'notes':{'visible_validation':chosen[name], 'selection':'finer single-layer ranks, >=0.5 point visible gain'}})
    # Test whether refined single-layer selection also supports further
    # compression across layers. Candidate pool is frozen by visible coarse
    # results; no hidden measurements affect this choice.
    chosen_layer=chosen['layer'] if chosen else refsel[name]['layer']
    chosen_ranks=chosen['ranks'] if chosen else refsel[name]['ranks']
    configs=[(chosen_layer,chosen_ranks)]
    current=chosen[name] if chosen else refsel[name]['validation']
    cache={}
    def rebuild(layer,ranks):
        key=(layer,tuple(ranks))
        if key not in cache:
            c,f,_=decompose_layer(model,layer,ranks,100,1e-8)
            cache[key]=overrides_for(layer,reconstruct_canonical(c,f))
        return cache[key]
    history=[]
    for step in range(2):
        base={}
        for layer,ranks in configs:
            base.update(rebuild(layer,ranks))
        winner=None
        for layer in range(12):
            if layer in {l for l,r in configs}:
                continue
            candidates=[e for e in entries if e['layer']==layer][:4]
            for entry in candidates:
                res=task.evaluate(model,{**base,**rebuild(layer,entry['ranks'])})
                if res['finite'] and (winner is None or (res['accuracy'],-res['loss'])>(winner[1]['accuracy'],-winner[1]['loss'])):
                    winner=(entry,res)
        history.append({'candidate':winner[0] if winner else None,'validation':winner[1] if winner else None})
        if winner is None or winner[1]['accuracy']<current['accuracy']+0.5:
            break
        configs.append((winner[0]['layer'],winner[0]['ranks']))
        current=winner[1]
    shutil.copytree(SUBS/'better_refined',SUBS/'better_refined_multi',dirs_exist_ok=True)
    write('better_refined_multi',name,{'format':'mha-structure-patch-v3','blocks':[patch_for(l,r) for l,r in configs],
          'notes':{'visible_validation':{k:current[k] for k in ('accuracy','loss')},'selection':'finer ranks then >=0.5 point greedy visible gains'}})
    save_json(OUT/'refined_greedy.json',{'configs':configs,'validation':{k:current[k] for k in ('accuracy','loss')},'history':history})
    # New procedure genuinely controls the result. Predeclared scalar grid,
    # selected on visible data, one layer with the reference ranks (no fitting).
    scales=(0.75,0.875,1.0,1.125,1.25,1.5)
    scale_log=load_json(OUT/'rescaling.json',{})
    for taskname,sel in refsel.items():
        visible=Task(taskname,device)
        layer,ranks=sel['layer'],sel['ranks']
        core,factors,_=decompose_layer(model,layer,ranks,100,1e-8)
        for scale in scales:
            key=f'{taskname}:{scale}'
            if key not in scale_log:
                rebuilt=reconstruct_canonical(core*scale,factors)
                res=visible.evaluate(model,overrides_for(layer,rebuilt))
                scale_log[key]={'scale':scale,'task':taskname, 'accuracy':res['accuracy'],'loss':res['loss']}
                save_json(OUT/'rescaling.json',scale_log)
        winner=max((v for v in scale_log.values() if v['task']==taskname),key=lambda v:(v['accuracy'],-v['loss']))
        for variant,scale in [('custom_equivalent',1.0),('variant_rescaled',winner['scale'])]:
            p={'format':'mha-structure-patch-v3','blocks':[{'layer':layer,'targets':list(MATRICES),
               'layout':'heads','procedure':'decompose.py','parameters':{'ranks':ranks,'scale':scale}}]}
            d=write(variant,taskname,p)
            shutil.copyfile(ROOT/'method/network_tucker.py',d/'decompose.py')
    # Submit every formerly ignored directory name as well.
    shutil.copytree(SUBS/'shortcut_missing_experiment',SUBS/'_shortcut_unlisted',dirs_exist_ok=True)
    print('Controls and visible-only variants ready',flush=True)

if __name__=='__main__':
    main()
