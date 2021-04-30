import numpy as np
from scipy.linalg import expm


class LieAlgebra:
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

            # Check for the zero vector
            if np.array_equal(u, np.zeros(3)):
                return v

            cot = lambda x: 1 / np.tan(x)  # noqa: E731

            alpha = np.linalg.norm(u)
            U = self.matrix(u)

            # Convert v from the matrix representation to the basis representation
            v = np.array([v[2, 1], v[0, 2], v[1, 0]])

            return (
                np.eye(3)
                - 0.5 * U
                - (2 - alpha * cot(0.5 * alpha)) / (2 * alpha ** 2) * U @ U
            ) @ v
        else:
            return super().dexpinv(u, v)

    def matrix(self, y):
        if y.size != 3:
            raise NotImplementedError("Not yet implemented for n != 3")
        u, v, w = y
        return np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
