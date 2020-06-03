from torch.nn import functional as F

from .module import Module
from .prng import rand


def get_coords(x, ndim):
    if isinstance(x, int):
        return (x,) * ndim

    if isinstance(x, (tuple, list)):
        assert len(x) == ndim
        return tuple(x)

    assert False


class ConvNd(Module):
    def __init__(self, in_channels, out_channels, face, stride, pad, ndim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.face = face
        self.stride = stride
        self.pad = pad
        self.ndim = ndim

        face = get_coords(face, ndim)
        shape = (in_channels, out_channels) + face
        x = rand(-0.05, 0.05, shape)
        x = x.transpose(0, 1)
        self.weight = self.parameter(x)

        self.bias = self.parameter(rand(-0.05, 0.05, (out_channels,)))

        self.conv = getattr(F, 'conv%dd' % ndim)

    def forward(self, x, is_t):
        return self.conv(x, self.weight, self.bias, self.stride, self.pad)


class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 1)


class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 2)


class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 3)
