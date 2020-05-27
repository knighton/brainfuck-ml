import numpy as np

from .module import Module
from .prng import rand


class Dropout(Module):
    def __init__(self, rate):
        assert 0 <= rate <= 1
        self.rate = rate

    def forward(self, x, is_t):
        if is_t:
            noise = rand(0, 1, x.shape)
            self.mask = (self.rate <= noise).astype(np.float32)
            x = x * self.mask / self.rate
        return x

    def backward(self, dy):
        return dy * self.mask / self.rate
