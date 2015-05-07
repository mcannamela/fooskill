from pymc import Deterministic


class DeterministicWithComputeValue(Deterministic):

    @classmethod
    def _compute_value(cls, **parent_values):
        raise NotImplementedError()

    def __init__(self, name, parents, doc='', **kwargs):
        super(DeterministicWithComputeValue, self).__init__(self._compute_value, doc, name, parents, **kwargs)