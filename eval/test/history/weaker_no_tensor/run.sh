#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/variants.py weaker_no_tensor --seed 0
