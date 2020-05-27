import numpy as np

from .module import Module


class Softmax(Module):
    def forward(self, x, is_t):
        x = x - x.max(1, keepdims=True)
        x = np.exp(x)
        return x / x.sum(1, keepdims=True)

    def backward(self, dy):
        return dy
