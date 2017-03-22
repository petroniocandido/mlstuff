#!/usr/bin/python
# -*- coding: utf8 -*-

from mlstuff.evolutionary.EvolutionaryAlgorithm import Variable
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA
import numpy as np


class SimpleRealGATest(SimpleRealGA):
    def __init__(self):
        super(SimpleRealGATest, self)\
            .__init__(objectives_number = 1, variables_number = 4, population_size = 20, max_generations = 50,
                      crossover_rate=0.8, crossover_alpha_pol=0.8, crossover_alpha = 0.5,
                      mutation_rate=0.4, mutation_rand = lambda : np.random.uniform(-5,5),
                      selection_probability=0.7, selection_elitism_rate=0.2)

        self.variables.append(Variable(name="x", type=int, pmin=-10,pmax=10))
        self.variables.append(Variable(name="y", type=int, pmin=-10, pmax=10))
        self.variables.append(Variable(name="k", type=int, pmin=1, pmax=10))
        self.variables.append(Variable(name="j", type=int, pmin=1, pmax=10))
        self.objectives.append(Variable(name="z", type=int, pmin=0, pmax=10))

    def evaluate_individual(self, individual):
        individual.objectives['z'] = (individual.variables['x'] ** 2 * individual.variables['y'] ** 2) / ( individual.variables['k'] * individual.variables['j'])
        individual.fitness = individual.objectives['z']

    def stop_criteria(self):
        return self.best_fitness == 0


test = SimpleRealGATest()

test.run()

for i in test.best_individuals:
    print(i)
