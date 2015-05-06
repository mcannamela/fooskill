from stochastics.stochastic_with_log_p import StochasticWithLogP
import numpy as np

class TalentPoolValueAccessor(object):

    @classmethod
    def get_mean_skill_mean(cls, value):
        return value[0]

    @classmethod
    def get_mean_skill_std_dev(cls, value):
        return value[1]

    @classmethod
    def get_log_of_std_dev_of_skill_mean(cls, value):
        return value[2]

    @classmethod
    def get_log_of_std_dev_of_skill_std_dev(cls, value):
        return value[3]

    @classmethod
    def build_value(cls,
                    mean_skill_mean,
                    mean_skill_std_dev,
                    log_of_std_dev_of_skill_mean,
                    log_of_std_dev_of_skill_std_dev):

        return np.array([mean_skill_mean,
                         mean_skill_std_dev,
                         log_of_std_dev_of_skill_mean,
                         log_of_std_dev_of_skill_std_dev], dtype=float)

SQRT_TWO_PI = (2*np.pi)**.5

def compute_gaussian_logp(x, mu, sigma):
    exp_arg = -(x-mu)**2 / (2*sigma**2)
    norm_term = -np.log(sigma*SQRT_TWO_PI)
    return exp_arg + norm_term

def draw_random_gaussian(mu, sigma):
    return mu + np.random.randn()*sigma


class TalentPool(StochasticWithLogP):


    PRIOR_MEAN = 0.0
    PRIOR_STD_DEV = 1.0

    @classmethod
    def _compute_logp(cls, value, **parent_values):

        # the mean skill is anchored, but we have flat priors over:
        # 1. the std dev of the mean skill. we don't know how by much variation in mean skill there is between individuals.
        # 2. the mean of the std dev of the skill. we don't know how variable an individual's performance is likely to be.
        # 3. the std dev of the std dev of the skill. we don't know how wild is the wildest individual or how consistent the most consistent
        mean_skill_mean = TalentPoolValueAccessor.get_mean_skill_mean(value)
        logp = compute_gaussian_logp(mean_skill_mean, cls.PRIOR_MEAN, cls.PRIOR_STD_DEV)

        return logp

    @classmethod
    def logp_of_mean_skill(self, value, mean_skill):
        mu = TalentPoolValueAccessor.get_mean_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_mean_skill_std_dev(value)
        return compute_gaussian_logp(mean_skill, mu, sigma)

    @classmethod
    def logp_of_std_dev_of_skill(self, value, std_dev_of_skill):
        mu = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_std_dev(value)
        return compute_gaussian_logp(np.log(std_dev_of_skill), mu, sigma)

    @classmethod
    def logp_of_skill_parameters(cls, value, mean_skill, std_dev_of_skill):
        return (cls.logp_of_mean_skill(value, mean_skill) +
                cls.logp_of_std_dev_of_skill(value, std_dev_of_skill))

    @classmethod
    def draw_random_mean_skill(cls, value):
        mu = TalentPoolValueAccessor.get_mean_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_mean_skill_std_dev(value)
        mean_skill_sample = draw_random_gaussian(mu, sigma)
        return mean_skill_sample

    @classmethod
    def draw_random_std_dev_of_skill(cls, value):
        mu = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_mean(value)
        sigma = TalentPoolValueAccessor.get_log_of_std_dev_of_skill_std_dev(value)
        log_of_std_dev_sample = draw_random_gaussian(mu, sigma)
        return np.exp(log_of_std_dev_sample)

    def __init__(self, name, **kwargs):
        self._ensure_initial_value_set(kwargs)
        parents = {}
        super(TalentPool, self).__init__(name, parents, **kwargs)

    def _ensure_initial_value_set(self, kwargs):
        initial_value = kwargs.get('value', None)
        if initial_value is not None:
            initial_value = np.asarray(initial_value, dtype=float)
            self._assert_value_valid(initial_value)
        else:
            initial_value = self._get_default_initial_value()
        kwargs['value'] = initial_value

    def _assert_value_valid(self, initial_value):
        if len(initial_value)!=4:
            raise ValueError("Value must have length of 4, not {}".format(len(initial_value)))

    def _get_default_initial_value(self):
        return TalentPoolValueAccessor.build_value(self.PRIOR_MEAN, 3.0, np.log(3.0), .2*np.log(3.0))










