# DiffPy

<p align="center">
A Python package for solving ordinary differential equations evolving on non-linear manifolds.
</p>

**Note: This package is still unfinished. The behaviour below is not yet implemented, but is a work in progress.**

This package is distributed with the [Python package index](https://pypi.org/). To install it, use

```bash
$ pip install diffpy
```

In order to solve an ODE, the differential must first be described in its canonical Lie form – that is, as a mapping from the manifold to the corresponding Lie algebra.
For examples, please see below.

## Equation evolving on the unit sphere

The unit sphere has Lie algebra _so_(3), consisting of 3⨉3 skew-symmetric matrices.
Ordinary differential equations where the solution space is the unit sphere may be formulated in the form

```
dy / dt = A(t, y) · y
```

where `A` is a skew-symmetric matrix.
In order to solve the above equation, you must define the function

```py
def A(t, y):
    # return a 3⨉3 skew-symmetric matrix of type np.ndarray
```

You must also decide which numerical scheme you would like to use to solve the equation.
Higher-order methods provide a more accurate solution, but are more computationally expensive.
For a list of available methods, see [available numerical schemes](#Available-numerical-schemes).
In this example, we will use the Lie group method corresponding to the fourth order Runge-Kutta method.

To solve the problem, we use the following code:

```py
import numpy as np
import diffpy

### Code defining or importing A(t, y) ###

y0 = [0.0, 0.0, 1.0]
t_start = 0
t_end = 5
step_length = 0.1
manifold = "hmnsphere"
method = "RKMK4"
solution = diffpy.solve(A, y0, t_start, t_end, step_length, manifold, method)
```
The variable `solution` is now a `Flow` object with two attributes: `T`, a one-dimensional numpy array containing the times at which the solution is computed, `Y`, a `3 ⨉ n` numpy array where column `Y[i, :]` is the solution at time `T[i]`.
It is also possible to use indexing directly on the object: `solution[i, j]` is equivalent to `solution.Y[i, j]`.

The following is a suggestion in order to plot the solution:

```py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(Y_diffpy[0, :], Y_diffpy[1, :], Y_diffpy[2, :])
plt.show()
```


## Available numerical schemes
