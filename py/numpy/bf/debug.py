from .module import Module


class Debug(Module):
    def __init__(self, text=None):
        self.text = text or ''

    def forward(self, x, is_t):
        print('%s %s' % (self.text, tuple(x.shape)))
        return x
