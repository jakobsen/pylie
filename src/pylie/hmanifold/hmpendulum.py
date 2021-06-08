import numpy as np
from .hmanifold import HomogenousManifold
from ..liealgebra import se_nLieAlgebra
from ..liegroup import SE_NLieGroup


class SphericalPendulum(HomogenousManifold):
    """The S2 sphere. Corresponding Lie group SO(n)."""

    def __init__(self, y=np.array([0, 0, 1])):
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except Exception:
                raise TypeError("y must be array_like")
        self.n = y.size
        self.y = y
        self.lie_group = SE_NLieGroup()
        self.lie_algebra = se_nLieAlgebra(self.lie_group)
        super().__init__()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value.size != self.n:
            raise ValueError("y does not have the correct dimension")
        self._y = value
