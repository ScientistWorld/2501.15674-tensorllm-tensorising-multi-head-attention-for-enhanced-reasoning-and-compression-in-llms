#!/bin/bash
# Install Python packages into the container overlay (system Python, no venv).
#
# PyTorch is the CUDA 12.6 build so that GPU jobs use the GPU; it also runs
# on CPU-only nodes.  pyarrow is needed only to rebuild the task data from the
# Bias in Bios parquet file.

set -e

uv pip install --system --reinstall-package torch "torch==2.12.0+cu126" \
    --index-url https://download.pytorch.org/whl/cu126

uv pip install --system \
    transformers \
    safetensors \
    tokenizers \
    numpy \
    pyarrow \
    tqdm
