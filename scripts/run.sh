#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

export TENSORLLM_LIMIT=80
export TENSORLLM_BATCH_SIZE=8
export TENSORLLM_LAYERS="0,6,11"
export TENSORLLM_RANKS="384,32,3 256,32,3"

bash scripts/method.sh
