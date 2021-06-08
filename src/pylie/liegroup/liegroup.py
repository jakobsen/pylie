import numpy as np


class LieGroup:
    def __init__(self):
        pass

    def action(self, g, u):
        raise NotImplementedError


class SOLieGroup(LieGroup):
    def action(self, g, u):
        return g @ u


class SELieGroup(LieGroup):
    def action(self, g, u):
        """The left (coadjoint) action of the
        group SE(3) on the dual of its lie algebra se(3)*

        Parameters
        ----------
        g : Two-tuple
            Element of the Lie group. First element is a 3x3 matrix,
            second element a 3-vector.
        u : array of length 6
            Element of the dual of se(3).

        Returns
        -------
        [type]
            [description]
        """
        G, g = g
        u, v = np.split(u, 2)
        z2 = G @ v
        z1 = G @ u + np.cross(g, z2)
        return np.hstack((z1, z2))


class SE_NLieGroup(SELieGroup):
    def action(self, g_arr, u):
        """[summary]

        Parameters
        ----------
        g : Array of two-tuples
            Elements of the Lie group.
        u : Array of length 6N
            Element of the manifold.
        """
        result = np.zeros(len(u))
        for i, g in enumerate(g_arr):
            result[6 * i : 6 * i + 6] = self._single_element_action(
                g, u[6 * i : 6 * i + 6]
            )
        return result

    def _single_element_action(self, g, u):
        """[summary]

        Parameters
        ----------
        g : Two-tuple
            First element 3-by-3 matrix, second element array of length 3
        u : Array
            Must be of length 3
        """
        if not isinstance(g, tuple):
            raise TypeError()
        if len(u) != 6:
            raise ValueError()

        g_matrix, g_vector = g
        q, omega = np.split(u, 2)
        new_q = g_matrix @ q
        new_omega = g_matrix @ omega + self._hat(g_vector) @ g_matrix @ q
        return np.hstack((new_q, new_omega))

    def _hat(self, y):
        if len(y) != 3:
            raise ValueError("y must be of length 3")
        x1, x2, x3 = y
        return np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0],])

