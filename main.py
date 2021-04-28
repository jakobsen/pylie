import diffpy
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa

if __name__ == "__main__":

    def f(t, y):
        L = np.array(
            [
                [0,               t,           -0.4 * np.cos(t)],
                [-t,              0,           0.1 * t],
                [0.4 * np.cos(t), -0.1 * t, 0]]
        )
        return L

    def f_scipy(t, y):
        return f(t, y) @ y

    y0 = np.array([0.0, 0.0, 1.0])
    flow = diffpy.solve(f, y0, 0, 5, 0.01, "hmnsphere", "E1")
    Y_scipy = solve_ivp(f_scipy, (0, 5), y0, t_eval=flow.T, rtol=1e-13, atol=1e-13).y

    ##### 3D BEGIN ##### noqa: E266
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
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="c", alpha=0.3, linewidth=0)
    ax.plot(flow[0, :], flow[1, :], flow[2, :])
    ax.plot(Y_scipy[0, :], Y_scipy[1, :], Y_scipy[2, :])

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
    ##### 3D END ##### noqa: E266

    # print(Y.shape)
    # print(Y_scipy.shape)
    # fig, ax = plt.subplots(dpi=150)
    # ax.plot(T, Y_diffpy[2, :])
    # ax.semilogy(T, [np.linalg.norm(Y_diffpy[:, i]) for i in range(501)])
    # ax.semilogy(T, [np.linalg.norm(Y_scipy[:, i]) for i in range(501)])
    plt.show()
