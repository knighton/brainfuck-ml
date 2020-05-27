import numpy as np


class Module(object):
    def each_parameter(self):
        yield from ()

    def zero_grad(self):
        for x, dx in self.each_parameter():
            dx -= dx

    def forward(self, x, is_t):
        return x

    def __call__(self, x, is_t):
        return self.forward(x, is_t)

    def backward(self, dy):
        return dy

    def update_step(self, lr):
        for x, dx in self.each_parameter():
            x -= lr * dx
