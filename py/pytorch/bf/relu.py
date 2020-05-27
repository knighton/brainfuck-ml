from .module import Module


class ReLU(Module):
    def forward(self, x, is_t):
        return x.clamp(min=0)
