import unittest
import pymc
from stochastics.talent_pool import TalentPool, TalentPoolValueAccessor

class TalentPoolTestCase(unittest.TestCase):

    POOL_NAME = 'the_talent_pool'

    def setUp(self):
        self._mean_skill_mean = 0.0
        self._mean_skill_std_dev = 1.0

        self._log_of_std_dev_of_skill_mean = 0.0
        self._log_of_std_dev_of_skill_std_dev = 1.0

    def _build_talent_pool_value(self):
        TalentPoolValueAccessor.build_value(self._mean_skill_mean,
                                            self._mean_skill_std_dev,
                                            self._log_of_std_dev_of_skill_mean,
                                            self._log_of_std_dev_of_skill_std_dev)
    def test_init(self):
        TalentPool(self.POOL_NAME)
        TalentPool(self.POOL_NAME,
                   value=self._build_talent_pool_value())

    def test_fit(self):
        talent_pool = TalentPool(self.POOL_NAME, self._build_talent_pool_value())

        mcmc = pymc.MCMC({talent_pool})

        mcmc.sample(100, 10, 2)






