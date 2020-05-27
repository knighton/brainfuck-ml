import torch

from ...common.bf import prng as lib


def set_seed(seed):
    lib.set_seed(seed)


def rand(low, high, shape):
    x = lib.rand(low, high, shape)
    return torch.tensor(x)


def randint(low, high, shape):
    x = lib.randint(low, high, shape)
    return torch.tensor(x, dtype=torch.int64)
