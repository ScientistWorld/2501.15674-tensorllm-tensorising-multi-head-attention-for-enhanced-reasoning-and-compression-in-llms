#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/shortcut_bruteforce/method.py
