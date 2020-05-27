from ...common.bf import prng as lib


def set_seed(seed):
    lib.set_seed(seed)


def rand(low, high, shape):
    return lib.rand(low, high, shape)


def randint(low, high, shape):
    return lib.randint(low, high, shape)
