#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/variants.py better_visible_search --seed 0
