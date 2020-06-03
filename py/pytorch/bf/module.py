class Module(object):
    def __init__(self):
        self.parameters = []

    def parameter(self, x):
        x.requires_grad_(True)
        self.parameters.append(x)
        return x

    def each_inner_parameter(self):
        yield from ()

    def each_parameter(self):
        for x in self.parameters:
            yield x
        for x in self.each_inner_parameter():
            yield x

    def zero_grad(self):
        for param in self.each_parameter():
            if param.grad is None:
                continue
            param.grad.zero_()

    def forward(self, x, is_t):
        return x

    def __call__(self, x, is_t):
        return self.forward(x, is_t)

    def update_step(self, lr):
        for param in self.each_parameter():
            if param.grad is None:
                continue
            param.data -= lr * param.grad
