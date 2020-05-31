import numpy as np

from .module import Module


mean = lambda a: a.mean(1, keepdims=True)


class BatchNormNd(Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5, ndim=None):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.ndim = ndim
        self.gamma = np.ones((dim, 1))
        self.dgamma = np.zeros((dim, 1))
        self.beta = np.zeros((dim, 1))
        self.dbeta = np.zeros((dim, 1))
        self.mov_mean = np.zeros((dim, 1))
        self.mov_std = np.zeros((dim, 1))
        self.x = None

    def each_parameter(self):
        yield self.gamma, self.dgamma
        yield self.beta, self.dbeta

    def forward(self, x, is_t):
        if self.ndim is not None:
            assert x.ndim == self.ndim + 2
        x = x.swapaxes(0, 1)
        s = x.shape
        x = x.reshape(x.shape[0], -1)
        if is_t:
            self.x = x
            x_mean = mean(x)
            x_ctr = x - x_mean
            x_var = mean(x_ctr ** 2)
            x_std = np.sqrt(x_var + self.eps)
            x_norm = x_ctr / x_std
            self.mov_mean = self.momentum * self.mov_mean + \
                (1 - self.momentum) * x_mean
            self.mov_std = self.momentum * self.mov_std + \
                (1 - self.momentum) * x_std
        else:
            x_norm = (x - self.mov_mean) / self.mov_std
        y = x_norm * self.gamma + self.beta
        y = y.reshape(*s)
        return y.swapaxes(0, 1)

    def backward(self, dy):
        dy = dy.swapaxes(0, 1)
        s = dy.shape
        dy = dy.reshape(dy.shape[0], -1)
        x_mean = mean(self.x)
        x_ctr = self.x - x_mean
        x_var = mean(x_ctr ** 2) + self.eps
        x_std = np.sqrt(x_var)
        x_norm = x_ctr / x_std
        self.dbeta = dy.sum(1, keepdims=True)
        self.dgamma = (dy * x_norm).sum(1, keepdims=True)
        dx = dy - mean(dy) - x_ctr / x_var * mean(dy * x_ctr)
        dx = dx * self.gamma / x_std
        dx = dx.reshape(*s)
        return dx.swapaxes(0, 1)


class BatchNorm0d(BatchNormNd):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__(dim, momentum, eps, 0)


class BatchNorm1d(BatchNormNd):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__(dim, momentum, eps, 1)


class BatchNorm2d(BatchNormNd):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__(dim, momentum, eps, 2)


class BatchNorm3d(BatchNormNd):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__(dim, momentum, eps, 3)
