#!/bin/bash
# Job entry point for hardening round 3: builds the WikidataQA splits, reproduces
# the reference end to end (scripts/reproduce.sh, inside the pipeline) and audits
# variants and shortcuts through the unchanged scorers.
set -e
cd "$(dirname "$0")/.."
bash redteam/hardening/run_round3.sh
