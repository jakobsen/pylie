import numpy as np


class TimeStepper:
    def __init__(self, exp, dexpinv, action):
        self.exp = exp
        self.dexpinv = dexpinv
        self.action = action
        self.a = None
        self.b = None
        self.c = None
        self.order = None
        self.s = None

    def step(self, f, t, y, h):
        n = y.size
        k_tilde = np.zeros((n, self.s))
        for i in range(self.s):
            u = np.zeros(n)
            for j in range(i - 1):
                # Explicit method, a is lower-triangular
                u += self.a[i, j] * k_tilde[:, j]
            k = f(t + self.c[i] * h, self.action(self.exp(h * u), y))
            k_tilde[:, i] = self.dexpinv(u, k, self.order)
        v = np.zeros(y.size)
        for i in range(self.s):
            v += self.b[i] * k_tilde[:, i]
        return self.action(self.exp(h * v), y)


class EulerLie(TimeStepper):
    def __init__(self, exp, dexpinv, action):
        super().__init__(exp, dexpinv, action)
        self.a = np.array([[0]])
        self.b = np.array([1])
        self.c = np.array([0])
        self.order = 1
        self.s = 1


class RKMK4(TimeStepper):
    def __init__(self, exp, dexpinv, action):
        super().__init__(exp, dexpinv, action)
        # fmt: off
        self.a = np.array(
            [
                [0,   0,   0,   0],
                [0.5, 0,   0,   0],
                [0,   0.5, 0,   0],
                [0,   0,   1.0, 0]
            ]
        )
        # fmt: on
        self.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        self.c = np.array([0, 0.5, 0.5, 1.0])
        self.order = 4
        self.s = 4
