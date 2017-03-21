#!/usr/bin/python
# -*- coding: utf8 -*-

from mlstuff.evolutionary.EvolutionaryAlgorithm import Variable
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA


class SimpleRealGATest(SimpleRealGA):
    def __init__(self):
        super(SimpleRealGATest, self)\
            .__init__(objectives_number = 1, variables_number = 2, population_size = 20,
                      max_generations = 50, mutation_rate=1)

        self.variables.append(Variable(name="x", type=int, pmin=-10,pmax=10))
        self.variables.append(Variable(name="y", type=int, pmin=-10, pmax=10))
        self.objectives.append(Variable(name="z", type=int, pmin=0, pmax=10))

    def evaluate_individual(self, individual):
        individual.objectives['z'] = individual.variables['x'] ** 2 * individual.variables['y'] ** 2
        individual.fitness = individual.objectives['z']

    def stop_criteria(self):
        return self.best_fitness == 0


test = SimpleRealGATest()

test.run()

for i in test.best_individuals:
    print(i)