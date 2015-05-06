import unittest
import pymc
from stochastics.player import Player, PlayerValueAccessor
from stochastics.talent_pool import TalentPool


class PlayerTestCase(unittest.TestCase):

    PLAYER_NAME = 'someplayer'

    def setUp(self):
        self._mean_firepower = 0.0
        self._std_dev_of_firepower = 1.0
        self._mean_stopping_power = 0.0
        self._std_dev_of_stopping_power = 1.0

        self._offensive_talent_pool = TalentPool('theoffense')
        self._defensive_talent_pool = TalentPool('thedefense')

    def _build_player_value(self):
        return PlayerValueAccessor.build_value(self._mean_firepower,
                    self._std_dev_of_firepower,
                    self._mean_stopping_power,
                    self._std_dev_of_stopping_power)
    def test_init(self):
        Player(self.PLAYER_NAME, self._offensive_talent_pool, self._defensive_talent_pool)
        Player(self.PLAYER_NAME, self._offensive_talent_pool, self._defensive_talent_pool,
                   value=self._build_player_value())

    def test_fit(self):
        player = Player(self.PLAYER_NAME,
                        self._offensive_talent_pool,
                        self._defensive_talent_pool,
                        value=self._build_player_value())
        mcmc = pymc.MCMC({player})

        mcmc.sample(100, 10, 2)






