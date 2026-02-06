import numpy as np


def default_rng(seed):
    return RNG(np.random.default_rng(seed))


class RNG:
    def __init__(self, rng):
        self.rng = rng

    def random(self, shape, out):
        self.rng.random(shape, out=out._data)
