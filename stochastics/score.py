from stochastics.stochastic_with_log_p import StochasticWithLogP
from scipy.stats import binom
import numpy as np

class ScoreValueAccessor(object):
    def get_team1_score(self, value):
        return value[0]

    def get_team2_score(self, value):
        return value[1]

    def get_total_score(self, value):
        return np.sum(value)

    def build_value(self, team1_score, team2_score):
        return np.array([team1_score, team2_score], dtype=float)


class Score(StochasticWithLogP):

    @classmethod
    def _compute_logp(cls, value, game=None):
        team1_score = ScoreValueAccessor.get_team1_score(value)
        total_score = ScoreValueAccessor.get_total_score(value)
        logp = binom.logpmf(team1_score, total_score, game)
        return logp

    def __init__(self, name, game, **kwargs):
        parents = {'game': game}
        super(Score, self).__init__(name, parents, **kwargs)


