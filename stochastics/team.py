import numpy as np
from pymc import Deterministic
from stochastics.player_performance import PlayerPerformanceValueAccessor


class TeamValueAccessor(object):

    @classmethod
    def get_offensive_firepower(cls, value):
        return value[0]

    @classmethod
    def get_defensive_firepower(cls, value):
        return value[1]

    @classmethod
    def get_stopping_power_vs_offense(cls, value):
        return value[2]

    @classmethod
    def get_stopping_power_vs_defense(cls, value):
        return value[3]

    @classmethod
    def build_value(cls,
                    offensive_firepower,
                    defensive_firepower,
                    stopping_power_vs_offense,
                    stopping_power_vs_defense):

        return np.array([offensive_firepower,
                         defensive_firepower,
                         stopping_power_vs_offense,
                         stopping_power_vs_defense], dtype=float)


class Team(Deterministic):

    @classmethod
    def _select_offensive_and_defensive_performance(cls, player1_on_offense, player1_performance, player2_performance):
        if player1_on_offense:
            offensive_performance = player1_performance
            defensive_performance = player2_performance
        else:
            offensive_performance = player2_performance
            defensive_performance = player1_performance

        return offensive_performance, defensive_performance

    @classmethod
    def _compute_value(cls, player1_performance, player2_performance, player1_on_offense):
        offensive_performance, defensive_performance = cls._select_offensive_and_defensive_performance(player1_on_offense,
                                                        player1_performance,
                                                        player2_performance)


        offensive_firepower = PlayerPerformanceValueAccessor.get_firepower(offensive_performance)
        defensive_firepower = PlayerPerformanceValueAccessor.get_firepower(defensive_performance)

        stopping_power_vs_offense = PlayerPerformanceValueAccessor.get_stopping_power(defensive_performance)
        stopping_power_vs_defense = stopping_power_vs_offense + PlayerPerformanceValueAccessor.get_stopping_power(offensive_performance)

        return TeamValueAccessor.build_value(offensive_firepower,
                                             defensive_firepower,
                                             stopping_power_vs_offense,
                                             stopping_power_vs_defense)

    def __init__(self, name, player1_performance, player2_performance, player1_on_offense, doc='', **kwargs):
        parents = {'player1_performance': player1_performance,
                   'player2_performance': player2_performance,
                   'player1_on_offense': player1_on_offense}
        super(Team, self).__init__(self._compute_value, doc, name, parents, **kwargs)




