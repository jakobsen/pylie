import numpy as np
from scipy.linalg import expm


class LieAlgebra:
    def __init__(self, LieGroup) -> None:
        self.action = LieGroup.action

    def exp(self, y):
        return expm(y)

    def dexpinv(self, u, v, order: int):
        ans = v
        if order >= 2:
            c = self.commutator(u, v)
            ans -= (1 / 2) * c
        if order >= 4:
            c = self.commutator(u, c)
            ans += (1 / 12) * c
        if order >= 6:
            raise NotImplementedError("Not yet implemented for order >= 6")
        return ans

    def commutator(self, a, b):
        if a.ndim == 2 and b.ndim == 2 and a.shape == b.shape:
            # a and b are matrices of similar dimensions
            return a @ b - b @ a
        else:
            raise NotImplementedError


class soLieAlgebra(LieAlgebra):
    def exp(self, y):
        if y.size == 3 and y.ndim == 1:
            # Use the rodrigues formula

            # Check if y is the zero vector
            if np.array_equal(y, np.zeros(3)):
                return np.eye(3)

            alpha = np.linalg.norm(y)
            Y = self.matrix(y)
            return (
                np.eye(3)
                + (np.sin(alpha) / alpha) * Y
                + ((1 - np.cos(alpha)) / alpha ** 2) * Y @ Y
            )
        # Otherwise, use the standard expm formula
        # We are here assuming that y is a matrix
        return super().exp(y)

    def dexpinv(self, u, v, _):
        if u.size == 3 and u.ndim == 1:
            # Use Rodrigues formula
            v_vector = np.array([v[2, 1], v[0, 2], v[1, 0]])
            alpha = np.linalg.norm(u)
            # Check for the zero vector
            if np.isclose(alpha, 0):
                return v_vector
            u_hat = self.matrix(u)
            lhs = (
                np.eye(3)
                - 0.5 * u_hat
                + (2 - alpha / np.tan(0.5 * alpha)) / (2 * alpha ** 2) * u_hat @ u_hat
            )
            return self.action(lhs, v_vector)
        else:
            return super().dexpinv(u, v)

    def matrix(self, y):
        if y.size != 3:
            raise NotImplementedError("Not yet implemented for n != 3")
        u, v, w = y
        return np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])


class seLieAlgebra(LieAlgebra):
    def _hat(self, y):
        u, v, w = y
        return np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])

    def _inv_hat(self, y):
        return np.array([y[2, 1], y[0, 2], y[1, 0]])

    def exp(self, y):
        u, v = np.split(y, 2)
        alpha = np.linalg.norm(u)
        u_hat = self._hat(u)
        if np.isclose(alpha, 0):
            u_exp = np.eye(3)
            v_exp = v
        else:
            u_exp = (
                np.eye(3)
                + (np.sin(alpha) / alpha) * u_hat
                + ((1 - np.cos(alpha)) / alpha ** 2) * u_hat @ u_hat
            )
            v_exp = (
                np.eye(3)
                + (1 - np.cos(alpha)) / (alpha ** 2) * u_hat
                + (alpha - np.sin(alpha)) / (alpha ** 3) * u_hat @ u_hat
            ) @ v
        return (u_exp, v_exp)

    def _cot(self, x):
        return 1 / np.tan(x)

    def _csc(self, x):
        return 1 / np.sin(x)

    def _dexpinv_helper_1(self, z):
        return (1 - 0.5 * z * self._cot(0.5 * z)) / (z ** 2)

    def _dexpinv_helper_2(self, z, rho):
        return (
            0.25
            * rho
            * ((z * self._csc(0.5 * z)) ** 2 + 2 * z * self._cot(0.5 * z) - 8)
            / (z ** 4)
        )

    def dexpinv(self, u, v, _=None):
        """Returns the result of dexp^(-1)_(u) (v).
        Both u and v are elements of se(3), represented as vectors of length 6"""
        A, a = np.split(u, 2)
        B, b = np.split(v, 2)
        alpha = np.linalg.norm(A)
        rho = np.inner(A, a)
        if np.isclose(alpha, 0):
            return v
        c1 = (
            B
            - 0.5 * np.cross(A, B)
            + self._dexpinv_helper_1(alpha) * np.cross(A, np.cross(A, B))
        )
        c2 = (
            b
            - 0.5 * (np.cross(a, B) + np.cross(A, b))
            + self._dexpinv_helper_2(alpha, rho) * np.cross(A, np.cross(A, B))
            + self._dexpinv_helper_1(alpha)
            * (
                np.cross(a, np.cross(A, B))
                + np.cross(A, np.cross(a, B))
                + np.cross(A, np.cross(A, b))
            )
        )
        return np.hstack((c1, c2))
