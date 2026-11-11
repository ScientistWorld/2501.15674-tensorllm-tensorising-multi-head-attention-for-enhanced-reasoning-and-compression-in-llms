#!/bin/bash
# Job entry point.  This iteration's job runs the hardening pipeline, which
# includes the full reference reproduction (scripts/reproduce.sh steps).
set -e
cd "$(dirname "$0")/.."
bash redteam/hardening/run.sh
