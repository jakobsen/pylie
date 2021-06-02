from ..solve import solve
from ..liegroup import SOLieGroup
from ..liealgebra import soLieAlgebra
import numpy as np
import unittest


def A(t, y):
    return np.array(
        [[0, t, -0.4 * np.cos(t)], [-t, 0, 0.1 * t], [0.4 * np.cos(t), -0.1 * t, 0]]
    )


class Testso3(unittest.TestCase):
    def test_matrix_representation(self):
        # Matrix-multiplication should be equal to the cross product
        so3 = soLieAlgebra(SOLieGroup())
        for _ in range(20):
            x = np.random.random(3)
            y = np.random.random(3)
            x_hat = so3.matrix(x)
            np.testing.assert_array_equal(x_hat @ y, np.cross(x, y))

    def test_solve(self):
        y0 = [0.0, 0.0, 1.0]
        t_start = 0
        t_end = 5
        step_length = 0.01
        manifold = "hmnsphere"
        method = "RKMK4"
        solution = solve(A, y0, t_start, t_end, step_length, manifold, method)
        for i in range(len(solution.T)):
            self.assertAlmostEqual(np.linalg.norm(solution[:, i]), 1.0)


if __name__ == "__main__":
    unittest.main()
