import torch

from .module import Module
from .prng import rand


class Dense(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.parameter(rand(-0.05, 0.05, (in_dim, out_dim)))
        self.bias = self.parameter(rand(-0.05, 0.05, (out_dim,)))

    def forward(self, x, is_t):
        return torch.einsum('ni,io->no', [x, self.weight]) + self.bias
