# PyLie

<p align="center">
A Python package for solving ordinary differential equations evolving on non-linear manifolds.
</p>

This package is distributed with the [Python package index](https://pypi.org/). To install it, use

```bash
$ pip install pylie
```

In order to solve an ODE, the differential must first be described in its canonical Lie form – that is, as a mapping from the manifold to the corresponding Lie algebra.
For examples, please see below.

## Example: Equation evolving on the unit sphere

_The complete code is listed at the bottom of this section if you want to copy-paste it, including a definition of `A(t, y)`._

The unit sphere has Lie algebra _so_(3), consisting of 3-by-3 skew-symmetric matrices (i.e. matrices which satisfy the equation `transpose(A) = -A`).
Ordinary differential equations where the solution space is the unit sphere may be formulated in the form

```
dy / dt = A(t, y) · y
```

where `A` is a skew-symmetric matrix.
In order to solve the above equation, you must define the function

```py
def A(t, y):
    # return a 3-by-3 skew-symmetric matrix of type np.ndarray
```

For instance:

```py
import numpy as np


def A(t, y):
    return np.array(
            [
                [0,                  t,           -0.4 * np.cos(t)],
                [-t,                 0,                0.1 * t    ],
                [0.4 * np.cos(t), -0.1 * t,                0      ]
            ]
        )
```

You must also decide which numerical scheme you would like to use to solve the equation.
Higher-order methods provide a more accurate solution, but are more computationally expensive.
For a list of available methods, see [available numerical schemes](#Available-numerical-schemes).
In this example, we will use the Lie group method corresponding to the fourth order Runge-Kutta method.

To solve the problem, we use the following code:

```py
import numpy as np
import pylie

### Code defining or importing A(t, y) ###

y0 = [0.0, 0.0, 1.0]
t_start = 0
t_end = 5
step_length = 0.1
manifold = "hmnsphere"
method = "RKMK4"
solution = pylie.solve(A, y0, t_start, t_end, step_length, manifold, method)
```

The variable `solution` is now a `Flow` object with two attributes: `T`, a one-dimensional numpy array containing the times at which the solution is computed, `Y`, a `3-by-n` numpy array where column `Y[i, :]` is the solution at time `T[i]`.
It is also possible to use indexing directly on the object: `solution[i, j]` is equivalent to `solution.Y[i, j]`.
If you wish you may also extract the variables `Y` and `T` directly by using

```py
# If solution is not yet computed:
Y, T = pylie.solve(A, y0, t_start, t_end, step_length, manifold, method)

# Or, if you followed the example above
Y, T = solution
```

The following is a suggestion in order to plot the solution:

```py
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(solution[0, :], solution[1, :], solution[2, :])
plt.show()
```

### Full example

This file is also avaiable in [`/docs/example.py`](/docs/example.py).

```py
from numpy.testing import assert_almost_equal
import pylie
import numpy as np
import matplotlib.pyplot as plt


def A(t, y):
    return np.array(
        [[0, t, -0.4 * np.cos(t)], [-t, 0, 0.1 * t], [0.4 * np.cos(t), -0.1 * t, 0]]
    )


if __name__ == "__main__":
    y0 = [0.0, 0.0, 1.0]
    t_start = 0
    t_end = 5
    step_length = 0.01
    manifold = "hmnsphere"
    method = "RKMK4"
    solution = pylie.solve(A, y0, t_start, t_end, step_length, manifold, method)

    # Verify that the solution is indeed on the unit sphere
    solution_norm = [np.linalg.norm(solution[:, i]) for i in range(len(solution.T))]
    for val in solution_norm:
        assert_almost_equal(val, 1.0)
    print("Passed test, plotting ...")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(solution[0, :], solution[1, :], solution[2, :])
    plt.show()

```

## Available numerical schemes

- `"E1"`: Explicit Euler, 1st order
- `"RKMK4"`: Runge-Kutta Munthe-Kaas 4, 4th order
