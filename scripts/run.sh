#!/bin/bash
# Job entry point: reproduce the reference end to end (GPU recommended; the
# layer x rank search takes about an hour on one A100 and is cached in
# outputs/tensorllm_search/).  The hardening audit that scores variants and
# shortcuts through the same scorers is redteam/hardening/run.sh.
set -e
cd "$(dirname "$0")/.."
bash scripts/reproduce.sh
