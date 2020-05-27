import numpy as np


rand_max = np.uint32(1 << 31)


class PRNG(object):
    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        raise NotImplementedError

    def get_value(self):
        raise NotImplementedError

    def get_tensor(self, *shape):
        size = np.prod(shape)
        x = np.empty(size, np.uint32)
        for i in range(size):
            x[i] = self.get_value()
        return x.reshape(*shape)


class Xorshift32(PRNG):
    def set_seed(self, seed):
        assert isinstance(seed, int)
        assert 0 <= seed < rand_max
        self.a = np.uint32(seed)

    def get_value(self):
        a = self.a
        a ^= a << np.uint32(13)
        a ^= a >> np.uint32(17)
        a ^= a << np.uint32(5)
        self.a = a
        return a % rand_max


class Distributions(object):
    def __init__(self, gen):
        self.gen = gen

    def rand(self, low, high, shape):
        assert low < high
        x = self.gen.get_tensor(*shape)
        x = x.astype(np.float32) / rand_max
        return x * (high - low) + low

    def randint(self, low, high, shape):
        assert low < high
        x = self.gen.get_tensor(*shape)
        x = x.astype(np.int32)
        return x % (high - low) + low


gen = Xorshift32(0x007)
dist = Distributions(gen)

set_seed = gen.set_seed
rand = dist.rand
randint = dist.randint
