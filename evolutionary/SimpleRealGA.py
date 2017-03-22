from mlstuff.evolutionary.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from mlstuff.evolutionary.operators import crossover,selection,mutation


class SimpleRealGA(EvolutionaryAlgorithm):
    def __init__(self, **kwargs):
        super(SimpleRealGA, self).__init__(**kwargs)

        self.operators.append(selection.Tournament(**kwargs))
        self.operators.append(crossover.SinglePointCrossover(**kwargs))
        self.operators.append(mutation.RealMutation(**kwargs))