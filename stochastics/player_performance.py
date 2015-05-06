from pymc import Stochastic
from stochastics.player import Player, PlayerValueAccessor
from stochastics.stochastic_with_log_p_and_random import StochasticWithLogPAndRandom
from stochastics.talent_pool import TalentPool, compute_gaussian_logp, TalentPoolValueAccessor, draw_random_gaussian
import numpy as np

class PlayerPerformanceValueAccessor(object):

    @classmethod
    def get_firepower(cls, value):
        return value[0]

    @classmethod
    def get_stopping_power(cls, value):
        return value[1]

    @classmethod
    def build_value(cls,
                    firepower,
                    stopping_power):

        return np.array([firepower,
                         stopping_power], dtype=float)


class PlayerPerformance(StochasticWithLogPAndRandom):
    
    @classmethod
    def _compute_logp(cls, value, player=None):
        firepower = PlayerPerformanceValueAccessor.get_firepower(value)
        stopping_power = PlayerPerformanceValueAccessor.get_stopping_power(value)

        logp = Player.compute_logp_of_performance(player, firepower, stopping_power)

        return logp

    @classmethod
    def _draw_random_sample(cls, player=None):
        mean_firepower = PlayerValueAccessor.get_mean_firepower(player)
        std_dev_of_firepower = PlayerValueAccessor.get_std_dev_of_firepower(player)

        mean_stopping_power = PlayerValueAccessor.get_mean_stopping_power(player)
        std_dev_of_stopping_power = PlayerValueAccessor.get_std_dev_of_stopping_power(player)

        firepower = draw_random_gaussian(mean_firepower, std_dev_of_firepower)
        stopping_power = draw_random_gaussian(mean_stopping_power, std_dev_of_stopping_power)

        return PlayerPerformanceValueAccessor.build_value(firepower, stopping_power)

    def __init__(self, name, player, **kwargs):
        parents = {'player': player}
        super(Player, self).__init__(name, parents, **kwargs)


