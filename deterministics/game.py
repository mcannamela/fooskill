from deterministics.deterministic_with_compute_value import DeterministicWithComputeValue
from stochastics.stochastic_with_log_p_and_random import StochasticWithLogPAndRandom
from stochastics.talent_pool import TalentPool, compute_gaussian_logp
import numpy as np

class Game(DeterministicWithComputeValue):

    @classmethod
    def _compute_value(cls, team1=None, team2=None):
        raise NotImplementedError()


    def __init__(self, name, team1, team2, **kwargs):
        parents = {'team1': team1,
                   'team2': team2}
        super(Game, self).__init__(name, parents, **kwargs)



