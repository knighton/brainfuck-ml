import tensorflow as tf

from .module import Module
from .prng import rand


def get_coords(x, ndim):
    if isinstance(x, int):
        return (x,) * ndim

    if isinstance(x, (tuple, list)):
        assert len(x) == ndim
        return tuple(x)

    assert False


def get_stride_ints(x, ndim):
    return [1] + list(get_coords(x, ndim)) + [1]


def get_pad_ints(x, ndim):
    pad = [[0, 0], [0, 0]]
    for a in get_coords(x, ndim):
        pad.append([a, a])
    return pad


class ConvNd(Module):
    def __init__(self, in_channels, out_channels, face, stride, pad, ndim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.face = face
        self.stride = stride
        self.stride_ints = get_stride_ints(stride, ndim)
        self.pad = pad
        self.pad_ints = get_pad_ints(pad, ndim)
        self.ndim = ndim

        face = get_coords(face, ndim)
        shape = (in_channels, out_channels) + face
        x = rand(-0.05, 0.05, shape)
        axes = list(range(2, ndim + 2)) + [0, 1]
        x = tf.transpose(x, axes)
        self.weight = self.parameter(x)

        self.bias = self.parameter(rand(-0.05, 0.05, (out_channels,)))

        self.conv = getattr(tf.nn, 'conv%dd' % ndim)

        self.channels_first = [0, ndim + 1] + list(range(1, ndim + 1))
        self.channels_last = [0] + list(range(2, ndim + 2)) + [1]

    def forward(self, x, is_t):
        x = tf.pad(x, self.pad_ints)
        x = tf.transpose(x, self.channels_last)
        x = self.conv(x, self.weight, self.stride_ints, 'VALID')
        x = tf.nn.bias_add(x, self.bias)
        return tf.transpose(x, self.channels_first)


class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 1)


class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 2)


class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 3)
