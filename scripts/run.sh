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

export TENSORLLM_LIMIT=320
export TENSORLLM_BATCH_SIZE=8
export TENSORLLM_LAYERS="3,4,5,6,7,8,9,10,11"
export TENSORLLM_RANKS="640,48,3 512,48,4 512,64,3 384,48,4 384,64,3 256,64,4"

bash scripts/method.sh
