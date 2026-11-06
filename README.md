# TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2501.15674"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2501.15674-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2501.15674](https://arxiv.org/abs/2501.15674) |
| **Domain** | Large language model compression and reasoning |
| **TL;DR** | TensorLLM applies shared-factor Tucker decomposition to multi-head attention weights after pretraining, aiming to denoise the model and compress attention parameters without fine-tuning. |

## Summary

The paper argues that existing post-training rank-reduction methods mainly benefit feed-forward layers and do not exploit the structure shared by attention heads. TensorLLM tensorizes the query, key, value, and output projections per head, then reconstructs them through a shared Tucker subspace across heads. Across RoBERTa, GPT-J, and LLaMA2 on four reasoning datasets, the paper reports accuracy gains together with MHA-layer compression.

## Problem

Can a fixed pretrained language model be post-processed so that its reasoning-task accuracy improves while the changed part of the model is represented with fewer parameters, without using additional training data or fine-tuning?

## The paper's method

TensorLLM is a post-training denoising and compression method for transformer multi-head attention weights. It targets the attention block rather than the feed-forward block and requires no extra data, training, or fine-tuning.

## Key results of the paper

- RoBERTa on FEVER improves from 50.0% to 50.45% accuracy with 3.74x MHA compression in the reported table.
- GPT-J shows larger FEVER gains, from 50.2% to 58.94% accuracy with 14.69x MHA compression.
- The largest reported MHA compression ratio is 247.3x on GPT-J HotPotQA.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +34.9 |
| `gpt-5.6-terra-xhigh` | -84.9 |
| `gpt-5.5-xhigh` | -89.9 |
| `gpt-5.6-luna-xhigh` | -92.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2501.15674 --mode innovate --agent claude-code
sw evaluate 2501.15674
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

