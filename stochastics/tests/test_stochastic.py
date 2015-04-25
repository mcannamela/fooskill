import unittest
import numpy as np
from pymc import Stochastic, MCMC
from pymc.examples import gelman_bioassay


class ParentStochastic(Stochastic):
    @classmethod
    def _my_logp(cls, value):
        return -np.sum(value**2)

    def __init__(self, name, doc='', **kwargs):
        parents = {}
        kwargs['value'] = kwargs.get('value', np.zeros(2))
        kwargs['dtype'] = float
        super(ParentStochastic, self).__init__(self._my_logp, doc, name, parents, **kwargs)


class MyStochastic(Stochastic):

    @classmethod
    def _my_logp(cls, value, some_parent=None):
        return -(value-np.sum(some_parent**2)**.5)**2

    @classmethod
    def _my_random(cls, some_parent=None):
        return np.random.randn(1)+np.sum(some_parent**2)**.5

    def __init__(self, name, parent, doc='', **kwargs):
        parents = {'some_parent': parent}
        kwargs['random'] = self._my_random
        kwargs['dtype'] = float
        super(MyStochastic, self).__init__(self._my_logp, doc, name, parents, **kwargs)


class StochasticTestCase(unittest.TestCase):

    PARENT_NAME = 'the_parent'
    STOCHASTIC_NAME = 'the_stochastic'
    SIBLING_NAME = 'sibling of stochastic'

    def test_init(self):
        MyStochastic(self.STOCHASTIC_NAME, 87)

    def _build_parent(self):
        p = ParentStochastic(self.PARENT_NAME)
        return p

    def test_init_with_parent(self):
        p = self._build_parent()
        s = MyStochastic(self.STOCHASTIC_NAME, p)

    def test_init_mcmc(self):
        p = self._build_parent()
        s = MyStochastic(self.STOCHASTIC_NAME, p)
        degenerately_named_s = MyStochastic(self.STOCHASTIC_NAME, p)
        sibling_s = MyStochastic(self.SIBLING_NAME, p)


        mcmc = MCMC({p, s})

        mcmc = MCMC({p, s, sibling_s})

        #all nodes must have unique name
        with self.assertRaises(ValueError):
            mcmc = MCMC({p, s, degenerately_named_s})


    def test_fit(self):
        p = self._build_parent()
        s = MyStochastic(self.STOCHASTIC_NAME, p)

        mcmc = MCMC({p, s})

        mcmc.sample(100, burn=10, thin=2)

    def test_fit_with_sibling(self):
        p = self._build_parent()
        s = MyStochastic(self.STOCHASTIC_NAME, p)
        sib = MyStochastic(self.SIBLING_NAME, p)

        mcmc = MCMC({p, s, sib})

        mcmc.sample(100, burn=10, thin=2)











