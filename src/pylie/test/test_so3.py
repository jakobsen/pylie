from ..solve import solve
import numpy as np
import unittest


def A(t, y):
    return np.array(
        [[0, t, -0.4 * np.cos(t)], [-t, 0, 0.1 * t], [0.4 * np.cos(t), -0.1 * t, 0]]
    )


class Testso3(unittest.TestCase):
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
