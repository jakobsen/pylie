import numpy as np
from collections.abc import Iterable
from typing import Callable

from ..hmanifold import HomogenousSphere, HeavyTop
from ..timestepper import EulerLie, ImprovedEulerLie, SSPRKMK3, RKMK4

_MANIFOLDS = {"hmnsphere": HomogenousSphere, "heavytop": HeavyTop}
_METHODS = {
    "E1": EulerLie,
    "E2": ImprovedEulerLie,
    "SSPRKMK3": SSPRKMK3,
    "RKMK4": RKMK4,
}


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
    timestepper = _METHODS[method](hmanifold)
    N_steps, last_step = divmod((t_end - t_start), h)
    N_steps = int(N_steps)
    T = [t_start + i * h for i in range(N_steps + 1)]
    number_of_cols = N_steps + 1 if np.isclose(last_step, 0) else N_steps + 2
    Y = np.zeros((len(y), number_of_cols))
    Y[:, 0] = y
    for i in range(1, N_steps + 1):
        # The y attribute of hmanifold is used to check against constraints
        # on elements of the given manifold
        # If anything fails here, it will raise an error
        hmanifold.y = timestepper.step(f, T[i - 1], hmanifold.y, h)
        Y[:, i] = hmanifold.y
    if not np.isclose(last_step, 0):
        hmanifold.y = timestepper.step(f, T[-1], hmanifold.y, last_step)
        Y[:, -1] = hmanifold.y
        T.append(t_end)
    return Flow(Y, T)
