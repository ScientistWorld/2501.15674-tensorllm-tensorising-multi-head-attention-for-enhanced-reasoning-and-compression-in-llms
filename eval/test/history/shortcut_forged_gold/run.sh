#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/harness.py shortcut_forged_gold
