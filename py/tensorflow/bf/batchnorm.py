import tensorflow as tf

from .module import Module


class BatchNormNd(Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5, ndim=None):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.ndim = ndim
        self.gamma = self.parameter(tf.ones((dim, 1)))
        self.beta = self.parameter(tf.zeros((dim, 1)))
        self.mov_mean = tf.zeros((dim, 1))
        self.mov_std = tf.ones((dim, 1))

    def forward(self, x, is_t):
        if self.ndim is not None:
            assert x.ndim == self.ndim + 2
        x = tf.einsum('ab...->ba...', x)
        s = x.shape
        x = tf.reshape(x, (x.shape[0], -1))
        if is_t:
            x_mean = tf.reduce_mean(x, 1, keepdims=True)
            x_ctr = x - x_mean
            x_var = tf.reduce_mean(x_ctr ** 2, 1, keepdims=True)
            x_std = tf.sqrt(x_var + self.eps)
            x_norm = x_ctr / x_std
            self.mov_mean = self.momentum * self.mov_mean + \
                (1 - self.momentum) * x_mean
            self.mov_std = self.momentum * self.mov_std + \
                (1 - self.momentum) * x_std
        else:
            x_norm = (x - self.mov_mean) / self.mov_std
        y = x_norm * self.gamma + self.beta
        y = tf.reshape(y, s)
        return tf.einsum('ab...->ba...', y)


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
