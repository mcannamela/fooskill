from pymc import Stochastic
import numpy as np

class TalentPoolValueAccessor(object):

    @classmethod
    def get_mean_skill_mean(self, value):
        return value[0]

    @classmethod
    def get_mean_skill_std_dev(self, value):
        return value[1]

    @classmethod
    def get_log_of_std_dev_of_skill_mean(self, value):
        return value[2]

    @classmethod
    def get_log_of_std_dev_of_skill_std_dev(self, value):
        return value[3]

SQRT_TWO_PI = (2*np.pi)**.5

def normal_logp(x, mu, sigma):
    exp_arg = -(x-mu)**2 / (2*sigma**2)
    norm_term = -np.log(sigma*SQRT_TWO_PI)
    return exp_arg + norm_term


class TalentPool(Stochastic):


    PRIOR_MEAN = 0.0

    @classmethod
    def _talent_pool_logp_fun(cls, value):

        mean_skill_mean = TalentPoolValueAccessor.get_mean_skill_mean(value)
        mean_skill_std_dev = TalentPoolValueAccessor.get_mean_skill_std_dev(value)

        #std_dev of skill doesn't come into play, meaning all variances are a-priori equally likely
        return normal_logp(mean_skill_mean, cls.PRIOR_MEAN, mean_skill_std_dev)

    @classmethod
    def logp_of_mean_skill(self, value, mean_skill):
        mu = TalentPoolValueAccessor.get_mean_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_mean_skill_std_dev(value)
        return normal_logp(mean_skill, mu, sigma)

    @classmethod
    def logp_of_std_dev_of_skill(self, value, std_dev_of_skill):
        mu = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_std_dev(value)
        return normal_logp(np.log(std_dev_of_skill), mu, sigma)

    def logp_of_skill_parameters(self, mean_skill, std_dev_of_skill):
        return (self.logp_of_mean_skill(self.value, mean_skill) +
                self.logp_of_std_dev_of_skill(self.value, std_dev_of_skill))

    def __init__(self, *args, **kwargs):
        kwargs['logp'] = self._talent_pool_logp_fun
        super(TalentPool, self).__init__(*args, **kwargs)








