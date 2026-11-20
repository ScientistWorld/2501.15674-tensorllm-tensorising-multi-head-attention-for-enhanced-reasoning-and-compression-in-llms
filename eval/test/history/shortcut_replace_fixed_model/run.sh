#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/harness.py shortcut_replace_fixed_model
