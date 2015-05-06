from deterministics.deterministic_with_compute_value import DeterministicWithComputeValue
import numpy as np
from deterministics.team import TeamValueAccessor


class Game(DeterministicWithComputeValue):

    @classmethod
    def _compute_value(cls, team1=None, team2=None):
        def get_o_f(t):
            return TeamValueAccessor.get_offensive_firepower(t)

        def get_s_vs_o(t):
            return TeamValueAccessor.get_stopping_power_vs_offense(t)

    @classmethod
    def _compute_probability_of_scoring(cls, firepower, stopping_power):
        arg = firepower-stopping_power
        e_arg = np.exp(arg)
        p = e_arg/(1.0+e_arg)
        return p


    def __init__(self, name, team1, team2, **kwargs):
        parents = {'team1': team1,
                   'team2': team2}
        super(Game, self).__init__(name, parents, **kwargs)



