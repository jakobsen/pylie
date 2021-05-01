import numpy as np
from .hmanifold import HomogenousManifold
from ..liealgebra import soLieAlgebra
from ..liegroup.liegroup import SOLieGroup


class HomogenousSphere(HomogenousManifold):
    def __init__(self, y):
        # Start with the unit vector pointing north
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except Exception:
                raise TypeError("y must be array_like")
        self.n = y.size
        self.y = y
        self.lie_group = SOLieGroup()
        self.lie_algebra = soLieAlgebra()
        super().__init__()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value.size != self.n:
            raise ValueError("y does not have the correct dimension")
        elif not np.isclose(np.linalg.norm(value), 1.0):
            raise ValueError(
                f"y does not lie on the N-sphere. ||y|| should be one, was {np.linalg.norm(value)}" # noqa
            )
        self._y = value
