from .module import Module


class Sequence(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def each_inner_parameter(self):
        for layer in self.layers:
            for param in layer.each_parameter():
                yield param

    def forward(self, x, is_t):
        for layer in self.layers:
            x = layer.forward(x, is_t)
        return x
