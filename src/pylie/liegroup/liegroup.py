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
