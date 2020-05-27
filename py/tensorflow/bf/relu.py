import tensorflow as tf

from .module import Module


class ReLU(Module):
    def forward(self, x, is_t):
        return tf.nn.relu(x)
