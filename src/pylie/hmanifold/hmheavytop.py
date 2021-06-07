import numpy as np
from ..liealgebra import seLieAlgebra
from ..liegroup import SELieGroup
from .hmanifold import HomogenousManifold


class HeavyTop(HomogenousManifold):
    """Manifold on which the Heavy Top equations evolve. Corresponding Lie group SE(3)."""
    def __init__(self, y=np.array([1, 0, 0, 1, 0, 0])):
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except Exception:
                raise TypeError("y must be array_like")
        self.n = y.size
        self.y = y
        self.lie_group = SELieGroup()
        self.lie_algebra = seLieAlgebra(self.lie_group)

        super().__init__()
