from pymc import Stochastic


class StochasticWithLogP(Stochastic):

    @classmethod
    def _compute_logp(cls, value, **parent_values):
        raise NotImplementedError()

    def __init__(self, name, parents, doc='', **kwargs):
        super(StochasticWithLogP, self).__init__(self._compute_logp, doc, name, parents, **kwargs)