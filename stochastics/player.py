from stochastics.stochastic_with_log_p_and_random import StochasticWithLogPAndRandom
from stochastics.talent_pool import TalentPool, compute_gaussian_logp
import numpy as np

class PlayerValueAccessor(object):

    @classmethod
    def get_mean_firepower(cls, value):
        return value[0]

    @classmethod
    def get_std_dev_of_firepower(cls, value):
        return value[1]

    @classmethod
    def get_mean_stopping_power(cls, value):
        return value[2]

    @classmethod
    def get_std_dev_of_stopping_power(cls, value):
        return value[3]

    @classmethod
    def build_value(cls,
                    mean_firepower,
                    std_dev_of_firepower,
                    mean_stopping_power,
                    std_dev_of_stopping_power):

        return np.array([mean_firepower,
                         std_dev_of_firepower,
                         mean_stopping_power,
                         std_dev_of_stopping_power], dtype=float)


class Player(StochasticWithLogPAndRandom):
    
    @classmethod
    def _compute_logp(cls, value, offensive_talent_pool=None, defensive_talent_pool=None):
        mean_firepower = PlayerValueAccessor.get_mean_firepower(value)
        std_dev_of_firepower = PlayerValueAccessor.get_std_dev_of_firepower(value)
        logp_firepower = TalentPool.logp_of_skill_parameters(offensive_talent_pool,
                                                             mean_firepower,
                                                             std_dev_of_firepower)

        mean_stopping_power = PlayerValueAccessor.get_mean_stopping_power(value)
        std_dev_of_stopping_power = PlayerValueAccessor.get_std_dev_of_stopping_power(value)
        logp_stopping_power = TalentPool.logp_of_skill_parameters(defensive_talent_pool,
                                                                  mean_stopping_power,
                                                                  std_dev_of_stopping_power)

        return logp_firepower + logp_stopping_power

    @classmethod
    def _draw_random_sample(cls, offensive_talent_pool=None, defensive_talent_pool=None):
        mean_firepower = TalentPool.draw_random_mean_skill(offensive_talent_pool)
        std_dev_of_firepower = TalentPool.draw_random_std_dev_of_skill(offensive_talent_pool)

        mean_stopping_power = TalentPool.draw_random_mean_skill(defensive_talent_pool)
        std_dev_of_stopping_power = TalentPool.draw_random_std_dev_of_skill(defensive_talent_pool)

        return PlayerValueAccessor.build_value(mean_firepower,
                                               std_dev_of_firepower,
                                               mean_stopping_power,
                                               std_dev_of_stopping_power)

    @classmethod
    def compute_logp_of_firepower(cls, value, firepower):
        return compute_gaussian_logp(firepower,
                           PlayerValueAccessor.get_mean_firepower(value),
                           PlayerValueAccessor.get_std_dev_of_firepower(value))

    @classmethod
    def compute_logp_of_stopping_power(cls, value, stopping_power):
        return compute_gaussian_logp(stopping_power,
                           PlayerValueAccessor.get_mean_stopping_power(value),
                           PlayerValueAccessor.get_std_dev_of_stopping_power(value))

    @classmethod
    def compute_logp_of_performance(cls, value, firepower, stopping_power):
        return (cls.compute_logp_of_firepower(value, firepower) +
                cls.compute_logp_of_stopping_power(value, stopping_power))

    def __init__(self, name, offensive_talent_pool, defensive_talent_pool, **kwargs):
        parents = {'offensive_talent_pool': offensive_talent_pool,
                   'defensive_talent_pool': defensive_talent_pool}
        super(Player, self).__init__(name, parents, **kwargs)


