class LieGroup:
    def __init__(self):
        pass

    def action(self, g, u):
        raise NotImplementedError


class SOLieGroup(LieGroup):
    def action(self, g, u):
        return g @ u
