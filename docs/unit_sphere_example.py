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
