from flow.flow import flow
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa

if __name__ == "__main__":

    def f(t, y):
        L = np.array(
            [[0, t, -0.4 * np.cos(t)], [-t, 0, 0.1 * t], [0.4 * np.cos(t), -0.1 * t, 0]]
        )
        return L @ y

    y0 = np.array([0.0, 0.0, 1.0])
    Y, T = flow(f, y0, 0, 5, 0.01, "hmnsphere", "E1")
    Y_scipy = odeint(f, y0, T, tfirst=True)
    # print(np.abs(Y - Y_scipy))

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi / 2:100j, 0.0:2.0 * pi:100j]
    X = r * sin(phi) * cos(theta)
    Y = r * sin(phi) * sin(theta)
    Z = r * cos(phi)

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="c", alpha=0.3, linewidth=0)
    ax.plot(Y[0, :], Y[1, :], Y[2, :], color="k")

    max_range = np.array(
        [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
    ).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        X.max() + X.min()
    )
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        Y.max() + Y.min()
    )
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        Z.max() + Z.min()
    )
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")
    plt.show()
