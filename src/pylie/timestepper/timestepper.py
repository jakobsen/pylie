import numpy as np


class TimeStepper:
    def __init__(self, manifold):
        self.exp = manifold.exp
        self.dexpinv = manifold.dexpinv
        self.action = manifold.action
        self.a = None
        self.b = None
        self.c = None
        self.order = None
        self.s = None

    def step(self, f, t, y, h):
        n = y.size
        k = np.zeros((n, self.s))
        for i in range(self.s):
            u = np.zeros(n)
            for j in range(i):
                u += self.a[i, j] * k[:, j]
            u *= h
            k[:, i] = self.dexpinv(
                u, f(t + self.c[i] * h, self.action(self.exp(u), y)), self.order
            )
        v = np.zeros(n)
        for i in range(self.s):
            v += self.b[i] * k[:, i]
        return self.action(self.exp(h * v), y)


class EulerLie(TimeStepper):
    def __init__(self, manifold):
        super().__init__(manifold)
        self.a = np.array([[0]])
        self.b = np.array([1])
        self.c = np.array([0])
        self.order = 1
        self.s = 1


class ImprovedEulerLie(TimeStepper):
    def __init__(self, manifold):
        super().__init__(manifold)
        self.a = np.array([[0, 0], [1, 0]])
        self.b = np.array([0.5, 0.5])
        self.c = np.array([0, 1])
        self.order = 2
        self.s = 2


class SSPRKMK3(TimeStepper):
    def __init__(self, manifold):
        super().__init__(manifold)
        self.a = np.array([[0, 0, 0], [1, 0, 0], [0.25, 0.25, 0]])
        self.b = np.array([1 / 6, 1 / 6, 2 / 3])
        self.c = np.array([0, 1, 0.5])
        self.order = 3
        self.s = 3


class RKMK4(TimeStepper):
    def __init__(self, manifold):
        super().__init__(manifold)
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
