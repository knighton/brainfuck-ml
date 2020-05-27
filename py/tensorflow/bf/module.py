import tensorflow as tf


class Module(object):
    def __init__(self):
        self.parameters = []

    def parameter(self, x):
        x = tf.Variable(x)
        self.parameters.append(x)
        return x

    def each_inner_parameter(self):
        yield from ()

    def each_parameter(self):
        for x in self.parameters:
            yield x
        for x in self.each_inner_parameter():
            yield x

    def forward(self, x, is_t):
        return x

    def __call__(self, x, is_t):
        return self.forward(x, is_t)
