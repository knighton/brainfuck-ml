import tensorflow as tf

from .module import Module


class Softmax(Module):
    def forward(self, x, is_t):
        x = x - tf.reduce_max(x, 1, keepdims=True)
        return tf.nn.softmax(x, 1)
