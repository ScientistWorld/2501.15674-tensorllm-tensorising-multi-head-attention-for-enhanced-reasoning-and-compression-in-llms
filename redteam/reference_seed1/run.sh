#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/variants.py reference_seed1 --seed 1
