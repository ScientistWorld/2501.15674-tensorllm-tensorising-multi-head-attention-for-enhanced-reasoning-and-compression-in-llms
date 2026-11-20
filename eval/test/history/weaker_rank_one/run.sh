#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/variants.py weaker_rank_one --seed 0
