from .module import Module


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x, is_t):
        s = (x.shape[0],) + self.shape
        return x.reshape(*s)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)
