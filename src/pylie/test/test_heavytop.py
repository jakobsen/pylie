from ..solve import solve
from ..liealgebra import seLieAlgebra
from ..liegroup import SELieGroup
import numpy as np
import unittest


def spinning_top(
    t, y, principal_moments=np.array([2, 2, 1]), m=1, g=1, chi=np.array([0, 0, 1])
):
    """A formulation of the problem exploiting the Lie-group structure"""
    mu, beta = np.split(y, 2)
    mu_dot = -mu / principal_moments
    beta_dot = -m * g * chi
    return np.hstack((mu_dot, beta_dot))


se3 = seLieAlgebra(SELieGroup())


class TestHeavyTop(unittest.TestCase):
    def test_solve(self):
        y0 = np.array([np.sin(1.1), 0, np.cos(1.1), 1, 0.2, 3])
        t_start = 0
        t_end = 5
        step_length = 0.01
        manifold = "heavytop"
        method = "RKMK4"
        solution = solve(
            spinning_top, y0, t_start, t_end, step_length, manifold, method
        )  # noqa: E501
        expected_norm = np.linalg.norm(y0[3:])
        for i in range(len(solution.T)):
            self.assertAlmostEqual(np.linalg.norm(solution[3:, i]), expected_norm)

    def test_dexpinv_zero(self):
        u = np.zeros(6)
        v = np.random.random(6)
        np.testing.assert_array_equal(se3.dexpinv(u, v), v)
        np.testing.assert_array_equal(se3.dexpinv(v, u), u)

    def test_dexpinv_nonzero(self):
        u = np.array([0.89120736, 0.0, 0.45359612, 1.0, 0.2, 3.0])
        v = np.array(
            [0.98688917, 0.62018318, 0.66178247, 0.67912063, 0.43851834, 0.8804615]
        )
        expected = np.array(
            [
                1.133009950431,
                0.638687002977,
                0.374690254641,
                1.618548639942,
                -0.730601007173,
                0.603188282585,
            ]
        )
        actual = se3.dexpinv(u, v)
        np.testing.assert_array_almost_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
