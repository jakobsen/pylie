class HomogenousManifold:
    """A homogenous manifold is a manifold acted upon by a Lie group action.

    This is the parent class of all homegenous manifolds.
    """

    def __init__(self, *args):
        self.lie_group = None
        self.lie_algebra = None
        self.exp = self.lie_algebra.exp
        self.dexp = self.lie_algebra.dexp
        self.action = self.lie_group.action

    def dist(self, a, b):
        # TODO
        pass

    def origin(self):
        # TODO
        pass

    def project(self, m):
        # TODO
        pass
