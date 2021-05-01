import numpy as np
from collections.abc import Iterable
from typing import Callable
from ..hmanifold import HomogenousSphere
from ..timestepper import EulerLie, RKMK4

_MANIFOLDS = {"hmnsphere": HomogenousSphere}
_METHODS = {"E1": EulerLie, "RKMK4": RKMK4}


class Flow:
    """Object which holds the calculated numerical approximation of an ODE.
    The attributes may be accessed by the regular dot syntax, or by
    ```
    Y, T = flow
    ```
    where flow is an instance of a Flow object.

    The object also supports indexing: `flow[i, j]` is equivalent
    to `flow.Y[i, j]`.

    Attributes
    ----------
    T : array
        List of values t at which the function y(t) is approximated
    Y : array
        Two-dimensional array containing numerical solution.
        Column Y[:, i] corresponds to the solution at T[i].
    """

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
    """Use the specified `method` to compute the numerical solution
    to the ODE defined by `f`. The return flow object will contain a
    parameter `Y` evolving on the given argument for `manifold`.

    To print supported methods and manifolds, use `pylie.manifolds()`
    and `pylie.methods()`.


    Parameters
    ----------
    f : Callable[[float, Iterable], Iterable]
        Function defining the differential equation.
        Must have call signature `f(t, y)`.
    y : Iterable
        Initial value
    t_start : number
        Initial time
    t_end : number
        End time
    h : number
        Step length
    manifold : str
        Manifold on which the ODE evolves. Must be one of
        the manifolds supported by `pylie`. Use `pylie.manifolds()`
        to print a list.
    method : str
        Method to use to solve the ODE. Use `pylie.methods()`
        to print a list of available methods.

    Returns
    -------
    Flow
        Flow object containing the solution, containing attributes Y and T.
        They may be accessed in one of the following ways:
        Either
        ```py
        flow = pylie.solve(*args)
        Y, T = flow
        ```
        or
        ```py
        flow = pylie.solve(*args)
        Y = flow.Y
        T = flow.T
        ```
        The flow object has the following attributes:
        T : array
            List of values t at which the function y(t) is approximated
        Y : array
            Two-dimensional array containing numerical solution.
            Column Y[:, i] corresponds to the solution at T[i].

        It also supports indexing: `flow[i, j]` is equivalent
        to `flow.Y[i, j]`.
    """
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
