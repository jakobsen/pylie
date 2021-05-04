from numpy.testing import assert_almost_equal
import pylie
import numpy as np
import matplotlib.pyplot as plt


def heavy_top(
    t, y, principal_moments=np.array([2, 2, 1]), m=1, g=1, chi=np.array([0, 0, 1])
):
    """A formulation of the problem exploiting the Lie-group structure"""
    mu, beta = np.split(y, 2)
    mu_dot = -mu / principal_moments
    beta_dot = -m * g * chi
    return np.hstack((mu_dot, beta_dot))


if __name__ == "__main__":
    y0 = np.array([np.sin(1.1), 0, np.cos(1.1), 1, 0.2, 3])
    t_start = 0
    t_end = 5
    step_length = 0.01
    manifold = "heavytop"
    method = "RKMK4"
    solution = pylie.solve(heavy_top, y0, t_start, t_end, step_length, manifold, method)

    # Verify that the solution is indeed on the manifold
    expexted_norm = np.linalg.norm(y0[3:])
    solution_norm = [np.linalg.norm(solution[3:, i]) for i in range(len(solution.T))]
    for val in solution_norm:
        assert_almost_equal(val, expexted_norm)
    print("Passed test, plotting ...")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(solution[0, :], solution[1, :], solution[2, :])
    plt.show()
