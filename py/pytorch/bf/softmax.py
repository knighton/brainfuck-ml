from .module import Module


class Softmax(Module):
    def forward(self, x, is_t):
        x = x - x.max(1, keepdim=True).values
        return x.softmax(1)
