#!/usr/bin/env python3
"""Final hidden-test verification and paired statistics. All raw data stays here."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'eval/test/private/final'
EVIDENCE=ROOT/'eval/test/final_results'
EXPERIMENTS=['bios_profession_roberta','fever_roberta_binary','wikidata_qa_roberta']
WORST={'accuracy':0.0,'loss':30.0,'mha_compression_ratio':1.0}


def load_module(side):
    spec=importlib.util.spec_from_file_location('audit_'+side,ROOT/f'eval/{side}/evaluate.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selftest():
    import torch
    PRIVATE.mkdir(parents=True,exist_ok=True)
    for side in ('train','test'):
        m=load_module(side)
        original=torch.arange(32,dtype=torch.float64).reshape(8,4)
        folder=PRIVATE/f'selftest_{side}'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'good.py').write_text('import torch\ndef decompose(x,p):\n u,s,v=torch.linalg.svd(x,full_matrices=False)\n return "ar,rb->ab", [u[:,:2]*s[:2],v[:2]]\n')
        rebuilt,error,count=m.procedure_reconstruction(folder,{'procedure':'good.py'},original)
        assert count==24 and torch.allclose(rebuilt,original,atol=1e-10)
        (folder/'read.py').write_text('def decompose(x,p):\n open("/home/user/eval/test/data/fever_test.jsonl").read()\n return "a->a",[x]\n')
        try:
            m.procedure_reconstruction(folder,{'procedure':'read.py'},original)
            raise AssertionError('confined method read test answers')
        except m.PatchError as exc:
            assert 'PermissionError' in str(exc),str(exc)
        (folder/'nan.py').write_text('def decompose(x,p):\n return "ar,rb->ab",[x[:,:1]*float("nan"),x[:1,:]]\n')
        try:
            m.procedure_reconstruction(folder,{'procedure':'nan.py'},original)
            raise AssertionError('NaN accepted')
        except m.PatchError:
            pass
    # Persist these attacks as normal submissions for the full hidden-test pass.
    for name,code in [('shortcut_read_labels','def decompose(x,p):\n open("/home/user/eval/test/data/fever_test.jsonl").read()\n return "a->a",[x]\n'),
                      ('shortcut_nan_network','def decompose(x,p):\n return "a,b->ab",[x.flatten()[:1]*float("nan"),x.flatten()[:1]]\n')]:
        for task in EXPERIMENTS:
            d=ROOT/'outputs/final_hardening/submissions'/name/task
            d.mkdir(parents=True,exist_ok=True)
            spec={'format':'mha-structure-patch-v3','blocks':[{'layer':0,'targets':['query'],'layout':'stack','procedure':'attack.py'}]}
            (d/'patch.json').write_text(json.dumps(spec))
            (d/'attack.py').write_text(code)
    print('Confinement, open-program reconstruction, and finite-value checks passed',flush=True)


def tamper():
    m=load_module('test')
    real=m.MODEL_DIR
    fake=PRIVATE/'tampered_model'
    fake.mkdir(parents=True,exist_ok=True)
    for item in real.iterdir():
        target=fake/item.name
        if target.exists():
            target.unlink()
        if item.name=='config.json':
            target.write_text('{}')
        else:
            target.symlink_to(item.resolve())
    m.MODEL_DIR=fake
    sys.argv=['evaluate.py','--output',str(PRIVATE/'tampered_scores.json'),'--key-sources']
    m.main()
    scores=json.loads((PRIVATE/'tampered_scores.json').read_text())
    assert all(v==WORST for e in scores['experiments'].values() for v in e['results'].values())


def paired(a,b):
    diffs=[float(x)-float(y) for x,y in zip(a,b)]
    n=len(diffs); mean=sum(diffs)/n
    se=math.sqrt(sum((x-mean)**2 for x in diffs)/(n*(n-1)))
    return {'n':n,'gain_points':100*mean,'se_points':100*se,
            'ci95_points':[100*(mean-1.96*se),100*(mean+1.96*se)],
            'z':mean/se if se else None}


def analyze():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    scores=json.loads((PRIVATE/'scores_test_all.json').read_text())
    train=json.loads((PRIVATE/'scores_train_all.json').read_text())
    ref=json.loads((ROOT/'scoring/reference.json').read_text())
    old=json.loads((ROOT/'eval/test/history/hardening/results_round3/scores_test_all.json').read_text())
    results={e:x['results'] for e,x in scores['experiments'].items()}
    methods=sorted(next(iter(results.values())))
    for e in EXPERIMENTS:
        assert set(results[e])==set(methods)
        for k,v in results[e].items():
            assert set(v)==set(WORST) and all(math.isfinite(x) for x in v.values())
            if scores['metadata']['methods'][k][e].get('run_failed'):
                assert all(results[t][k]==WORST for t in EXPERIMENTS),(k,e)
    def ranks(data,cohort):
        scales={}
        for e in EXPERIMENTS:
            for metric in WORST:
                vals=[data[e][k][metric] for k in cohort]
                avg=sum(vals)/len(vals)
                scales[e,metric]=math.sqrt(sum((x-avg)**2 for x in vals)/len(vals))
        rank={}
        for key in cohort:
            value=0.
            for e,exp in ref['experiments'].items():
                for metric,info in exp['metrics'].items():
                    sd=scales[e,metric]
                    if sd:
                        direction=1 if info['higher_is_better'] else -1
                        value+=exp['weight']*abs(info['coefficient'])*direction*(data[e][key][metric]-data[e]['tensorllm_roberta'][metric])/sd
            rank[key]=value
        return rank,scales
    rank,scales=ranks(results,methods)
    common=sorted(set(methods)&set(old['experiments'][EXPERIMENTS[0]]['results']))
    previous={e:old['experiments'][e]['results'] for e in EXPERIMENTS}
    oldrank,_=ranks(previous,common)
    new_common,_=ranks(results,common)
    pairs={}; rank_noise={}
    dumps=PRIVATE/'dumps_test'
    for key in methods:
        if all(scores['metadata']['methods'][key][e].get('status')=='ok' for e in EXPERIMENTS):
            pairs[key]={}
            variance=0.
            for e,exp in ref['experiments'].items():
                pa=dumps/e/f'{key}.json'; pb=dumps/e/'tensorllm_roberta.json'
                a=json.loads(pa.read_text()); b=json.loads(pb.read_text())
                pairs[key][e]=paired(a['correct'],b['correct'])
                weights={metric:exp['weight']*abs(info['coefficient'])*(1 if info['higher_is_better'] else -1)/scales[e,metric]
                         if scales[e,metric] else 0. for metric,info in exp['metrics'].items()}
                values=[100*weights['accuracy']*(x-y)+weights['loss']*(min(30.,l)-min(30.,r))
                        for x,y,l,r in zip(a['correct'],b['correct'],a['nll'],b['nll'])]
                n=len(values);mean=sum(values)/n
                variance+=sum((x-mean)**2 for x in values)/(n*(n-1))
            se=math.sqrt(variance)
            rank_noise[key]={'rank_gain':rank[key],'paired_se':se,'ci95':[rank[key]-1.96*se,rank[key]+1.96*se]}
    # Reference predictions are deterministic on an exclusive GPU run; this
    # comparison checks that adding methods to a score pass changes no numbers.
    alone=json.loads((PRIVATE/'scores_test_reference.json').read_text())
    for e in EXPERIMENTS:
        for k in ('roberta_original','tensorllm_roberta'):
            assert alone['experiments'][e]['results'][k]==results[e][k],(e,k,'batch/cohort regression')
    summary={'ranking_definition':'sum experiment weight * abs(coefficient) * oriented improvement / population SD across listed methods',
             'cohort':methods,'rank':rank,'paired_comparisons':pairs,'paired_rank_noise':rank_noise,
             'before_rank_common_cohort':oldrank,'after_rank_common_cohort':new_common,
             'raw_test_accuracy':{k:{e:scores['metadata']['methods'][k][e].get('raw_accuracy') for e in EXPERIMENTS} for k in methods},
             'reference_unchanged_between_exclusive_and_audit_runs':True,
             'all_failed_runs_have_complete_worst_values':True}
    (EVIDENCE/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ('scores_test_all.json','scores_train_all.json','tampered_scores.json','scores_test_reference.json','scores_train_reference.json'):
        shutil.copyfile(PRIVATE/name,EVIDENCE/name)
    shutil.copytree(ROOT/'outputs/final_hardening/submissions',EVIDENCE/'patches',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.safetensors','*.pt'))
    shutil.copyfile(ROOT/'outputs/final_hardening/refinement.json',EVIDENCE/'refinement.json')
    shutil.copyfile(ROOT/'outputs/final_hardening/rescaling.json',EVIDENCE/'rescaling.json')
    print(json.dumps({'rank':rank,'paired_rank_noise':rank_noise},indent=2),flush=True)

if __name__=='__main__':
    {'selftest':selftest,'tamper':tamper,'analyze':analyze}[sys.argv[1]]()
