import tensorflow as tf

from ...common.bf import prng as lib


def set_seed(seed):
    lib.set_seed(seed)


def rand(low, high, shape):
    x = lib.rand(low, high, shape)
    return tf.convert_to_tensor(x)


def randint(low, high, shape):
    x = lib.randint(low, high, shape)
    return tf.convert_to_tensor(x, tf.int64)
