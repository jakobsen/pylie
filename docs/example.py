import diffpy
import numpy as np
import matplotlib.pyplot as plt


def A(t, y):
    return np.array(
            [
                [0,                  t,           -0.4 * np.cos(t)],
                [-t,                 0,                0.1 * t    ],
                [0.4 * np.cos(t), -0.1 * t,                0      ]
            ]
        )


if __name__ == "__main__":
    y0 = [0.0, 0.0, 1.0]
    t_start = 0
    t_end = 5
    step_length = 0.01
    manifold = "hmnsphere"
    method = "RKMK4"
    solution = diffpy.solve(A, y0, t_start, t_end, step_length, manifold, method)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(solution[0, :], solution[1, :], solution[2, :])
    plt.show()
