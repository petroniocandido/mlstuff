
import mlstuff.evolutionary.EvolutionaryAlgorithm
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA


class SimpleRealGATest(SimpleRealGA):
    def __init__(self):
        super(SimpleRealGATest, self)\
            .__init__(objectives_number = 1, variables_number = 1, population_size = 20,
                      max_generations = 50)

        self.variables.append(Variable(name="x", type=int, pmin=-10,pmax=10))
        self.objectives.append(Variable(name="y", type=int, pmin=0, pmax=10))

    def evaluate_individual(self, individual):
        individual.objectives['y'] = individual.variables['x'] ** 2
        individual.fitness = individual.objectives['y']

    def stop_criteria(self):
        return self.best_fitness == 0


test = SimpleRealGATest()

test.run()