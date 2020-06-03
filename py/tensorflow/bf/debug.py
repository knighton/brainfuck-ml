import tensorflow as tf

from .module import Module


class Debug(Module):
    def __init__(self, text=None):
        super().__init__()
        self.text = text or ''

    def forward(self, x, is_t):
        m = tf.math.reduce_mean(x)
        s = tf.math.reduce_std(x)
        print('%s %s %.6f %.6f' % (self.text, tuple(x.shape), m, s))
        return x
