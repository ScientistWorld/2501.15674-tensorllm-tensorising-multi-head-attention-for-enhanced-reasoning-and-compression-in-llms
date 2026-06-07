# Problem Definition

## Research Question

Can a pretrained transformer language model be post-processed so that its reasoning-task accuracy improves while its attention-block parameters are compressed, without using additional training data or fine-tuning?

## Why It Matters

Large language models contain many redundant or noisy parameters. A good post-training procedure would make existing models smaller and, ideally, more accurate on reasoning tasks, without the cost and risk of retraining.

## Success Criteria

A successful solution should improve task accuracy over the original pretrained model on a fixed reasoning benchmark, avoid a large increase in prediction loss, and reduce the parameter count needed to represent the modified attention layer.
