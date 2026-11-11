#!/bin/bash
# Job entry point for hardening round 2: self-test of the scorer changes, the
# reference reproduced end to end from the final code (scripts/reproduce.sh,
# layer x rank search cached in outputs/tensorllm_search/), round-2 variants and
# shortcuts scored through the unchanged scorers, and the analysis.
set -e
cd "$(dirname "$0")/.."
bash redteam/hardening/run_round2.sh
