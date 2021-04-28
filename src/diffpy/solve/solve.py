import numpy as np
from collections.abc import Iterable
from typing import Callable
from ..hmanifold import HomogenousSphere
from ..timestepper import EulerLie, RKMK4

_MANIFOLDS = {"hmnsphere": HomogenousSphere}
_METHODS = {"E1": EulerLie, "RKMK4": RKMK4}


class Flow:
    def __init__(self, Y, T):
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if not isinstance(T, Iterable):
            raise TypeError("T must be array-like")
        self.Y = Y
        self.T = T

    def __iter__(self):
        yield from (self.Y.transpose(), self.T)

    def __getitem__(self, key):
        return self.Y[key]


def solve(
    f: Callable[[float, Iterable], Iterable],
    y,
    t_start,
    t_end,
    h,
    manifold: str,
    method: str,
):
    hmanifold = _MANIFOLDS[manifold](y)
    timestepper = _METHODS[method](hmanifold.exp, hmanifold.dexpinv, hmanifold.action)
    t = t_start
    T = [t]
    Y = np.array(y)

    while not np.isclose(t_end - t, 0):
        h = min(h, t_end - t)
        hmanifold.y = timestepper.step(f, t, hmanifold.y, h)
        t = t + h
        Y = np.column_stack((Y, hmanifold.y))
        T.append(t)

    return Flow(Y, np.array(T))
