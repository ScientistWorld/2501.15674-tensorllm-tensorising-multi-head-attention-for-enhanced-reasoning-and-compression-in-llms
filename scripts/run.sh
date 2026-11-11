#!/bin/bash
# Job entry point: reproduce the reference end to end (GPU recommended; the
# layer x rank search takes about an hour on one A100 and is cached in
# outputs/tensorllm_search/).  The hardening audits that score variants and
# shortcuts through the same scorers are redteam/hardening/run.sh (round 1)
# and redteam/hardening/run_round2.sh (round 2).
set -e
cd "$(dirname "$0")/.."
bash scripts/reproduce.sh
