from pymc import Stochastic
class Team(Stochastic):
    def __init__(self, player1, player2):
        self.players = [player1, player2]


