"""Example open-interface program: data-free HOOI with optional core rescaling.

The scalar is a predeclared hyperparameter, never fitted to labels. This file
is copied into a submission and run by the evaluator with original weights.
It is not the reference method (which uses the paper's unscaled HOOI).
"""
import math
import torch


def dot(x, a, mode):
    y = x.movedim(mode, 0)
    return (a @ y.reshape(y.shape[0], -1)).reshape((a.shape[0],) + y.shape[1:]).movedim(0, mode)


def top(x, mode, rank):
    x = x.movedim(mode, 0).reshape(x.shape[mode], -1)
    return torch.linalg.eigh(x @ x.T)[1][:, -rank:].flip(-1).contiguous()


def decompose(original, parameters):
    x = original.double()
    ranks = parameters['ranks']
    factors = [top(x, i, r) for i, r in enumerate(ranks)]
    norm = float((x*x).sum())
    last = None
    for iteration in range(100):
        for i, rank in enumerate(ranks):
            projected = x
            for j, u in enumerate(factors):
                if j != i:
                    projected = dot(projected, u.T, j)
            factors[i] = top(projected, i, rank)
        core = x
        for i,u in enumerate(factors):
            core = dot(core, u.T, i)
        error = math.sqrt(abs(norm - float((core*core).sum())) / norm)
        if iteration > 1 and abs(last-error) < 1e-8:
            break
        last = error
    core = core * parameters.get('scale', 1.0)
    return 'abch,ma,db,tc->mdth', [core, *factors]
