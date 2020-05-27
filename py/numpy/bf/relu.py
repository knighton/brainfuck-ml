from .module import Module


class ReLU(Module):
    def __init__(self):
        self.x = None

    def forward(self, x, is_t):
        if is_t:
            self.x = x
        return x.clip(min=0)

    def backward(self, dy):
        dx = dy.copy()
        dx[self.x < 0] = 0
        return dx
