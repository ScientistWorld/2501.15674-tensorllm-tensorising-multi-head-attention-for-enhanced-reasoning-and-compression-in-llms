# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json`
- [ ] Workspace validated via `python validate.py`
- [ ] `scoring/scores.json`
- [x] `scoring/EXPERIMENTS.md`
- [x] `scoring/TARGETS.md`
- [x] `scoring/CONSTRAINTS.md`
- [x] `scoring/DIRECTION.md`

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [x] Baseline script for original model condition

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh`
- [x] `scripts/baseline.sh`

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Dataset acquired
- [x] Pretrained model downloaded

## Milestones
- [ ] `method_runs`
- [ ] `core_claim`
- [ ] `core_claim_plus`
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
