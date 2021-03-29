import numpy as np
from typing import Callable, Iterable
from hmanifold.hmnsphere import HomogenousSphere
from timestepper.timestepper import EulerLie, RKMK4

MANIFOLDS = {"hmnsphere": HomogenousSphere}

METHODS = {"E1": EulerLie, "RKMK4": RKMK4}


def flow(
    f: Callable[[float, Iterable], Iterable],
    y,
    t_start,
    t_end,
    h,
    manifold: str,
    method: str,
):
    hmanifold = MANIFOLDS[manifold](y)
    timestepper = METHODS[method](hmanifold.exp, hmanifold.dexpinv, hmanifold.action)
    t = t_start
    T = [t]
    Y = np.array(y)

    while not np.isclose(t_end - t, 0):
        h = min(h, t_end - t_start)
        hmanifold.y = timestepper.step(f, t, hmanifold.y, h)
        t = t + h
        Y = np.column_stack((Y, hmanifold.y))
        T.append(t)

    return np.transpose(Y), np.array(T)
