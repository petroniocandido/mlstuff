
from evolutionary import EvolutionaryAlgorithm
from evolutionary.operators import crossover,selection,mutation


class SimpleRealGA(EvolutionaryAlgorithm.EvolutionaryAlgorithm):
    def __init__(self, **kwargs):
        super(SimpleRealGA, self).__init__(kwargs)

        self.operators.append(selection.Tournament(kwargs))
        self.operators.append(crossover.SinglePointCrossover(kwargs))
        self.operators.append(mutation.GaussRandomMutation(kwargs))