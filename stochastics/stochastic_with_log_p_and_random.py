from stochastics.stochastic_with_log_p import StochasticWithLogP

class StochasticWithLogPAndRandom(StochasticWithLogP):

    @classmethod
    def _draw_random_sample(cls, **parent_values):
        raise NotImplementedError()


    def __init__(self, *args, **kwargs):
        kwargs['random'] = self._draw_random_sample
        super(StochasticWithLogPAndRandom, self).__init__(*args, **kwargs)