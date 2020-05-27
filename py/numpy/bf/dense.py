import numpy as np

from .module import Module
from .prng import rand


class Dense(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = rand(-0.05, 0.05, (in_dim, out_dim))
        self.dweight = np.zeros_like(self.weight)
        self.bias = rand(-0.05, 0.05, (out_dim,))
        self.dbias = np.zeros_like(self.bias)

    def each_parameter(self):
        yield self.weight, self.dweight
        yield self.bias, self.dbias

    def forward(self, x, is_t):
        if is_t:
            self.x = x
        return np.einsum('ni,io->no', x, self.weight) + self.bias

    def backward(self, dy):
        self.dweight = np.einsum('ni,no->io', self.x, dy)
        self.dbias = dy.sum(0)
        return np.einsum('no,io->ni', dy, self.weight)
