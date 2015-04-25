from pymc import Stochastic
from stochastics.talent_pool import TalentPool, normal_logp
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


class Player(Stochastic):
    
    @classmethod
    def _player_logp_fun(cls, value, offensive_talent_pool=TalentPool(), defensive_talent_pool=TalentPool()):
        mean_firepower = PlayerValueAccessor.get_mean_firepower(value)
        std_dev_of_firepower = PlayerValueAccessor.get_std_dev_of_firepower(value)
        logp_firepower = offensive_talent_pool.logp_of_skill_parameters(mean_firepower, std_dev_of_firepower)

        mean_stopping_power = PlayerValueAccessor.get_mean_stopping_power(value)
        std_dev_of_stopping_power = PlayerValueAccessor.get_std_dev_of_stopping_power(value)
        logp_stopping_power = defensive_talent_pool.logp_of_skill_parameters(mean_stopping_power, std_dev_of_stopping_power)

        return logp_firepower + logp_stopping_power

    @classmethod
    def logp_of_firepower(cls, value, firepower):
        return normal_logp(firepower,
                           PlayerValueAccessor.get_mean_firepower(value),
                           PlayerValueAccessor.get_std_dev_of_firepower(value))

    @classmethod
    def logp_of_stopping_power(cls, value, stopping_power):
        return normal_logp(stopping_power,
                           PlayerValueAccessor.get_mean_stopping_power(value),
                           PlayerValueAccessor.get_std_dev_of_stopping_power(value))

    def logp_of_performance(self, firepower, stopping_power):
        return (self.logp_of_firepower(self.value, firepower) +
                self.logp_of_stopping_power(self.value, stopping_power))

    def __init__(self, *args, **kwargs):
        kwargs['logp'] = self._player_logp_fun
        super(Player, self).__init__(*args, **kwargs)

