import tensorflow as tf

from .module import Module
from .prng import rand


class Dropout(Module):
    def __init__(self, rate):
        super().__init__()
        assert 0 <= rate <= 1
        self.rate = rate

    def forward(self, x, is_t):
        if is_t:
            noise = rand(0, 1, x.shape)
            mask = tf.cast(self.rate <= noise, tf.float32)
            x = mask * x / self.rate
        return x
