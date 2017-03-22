from numpy import random
import numpy as np
from copy import deepcopy


class RouletteWheel(object):
    def __init__(self,**kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)

    def process(self,context):
        num_individuals = int(context.population_size * self.selection_rate)

        individuals = []

        for count in range(0, num_individuals):
            pass


class Tournament(object):
    def __init__(self, **kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)
        self.selection_probability = kwargs.get('selection_probability', 0.7)
        self.selection_elitism_rate = kwargs.get('selection_elitism_rate', 0.1)
        self.tournament_enable = kwargs.get('tournament_enable', lambda c: True)

    def process(self, context):

        if not self.tournament_enable(context):
            return

        elit = int(context.population_size * self.selection_elitism_rate)
        num_individuals = int(context.population_size * self.selection_rate)

        individuals = []

        for count in np.arange(0, elit):
            individuals.append(context.population[count])

        for count in np.arange(elit, num_individuals):
            tmp1 = random.randint(elit, context.population_size)
            tmp2 = random.randint(elit, context.population_size)

            while tmp1 == tmp2:
                tmp2 = random.randint(elit, context.population_size)

            individual_best = context.population[tmp1] if context.population[tmp1].fitness <= context.population[tmp2].fitness \
                else context.population[tmp2]
            individual_worst = context.population[tmp2] if individual_best.fitness == context.population[tmp1].fitness \
                else context.population[tmp1]

            p = random.uniform(0,1)

            if p <= self.selection_probability:
                individuals.append(individual_best)
            else:
                individuals.append(individual_worst)

        context.population = individuals


class Elitism(object):
    def __init__(self, **kwargs):
        self.elitism_enable = kwargs.get('elitism_enable', lambda c: True)

    def process(self, context):

        if not self.elitism_enable(context):
            return

        context.population = sorted(context.population, key=lambda i: i.fitness)

        if context.best_individuals_size > 1:
            context.population[-1 : -context.best_individuals_size-1] = deepcopy(context.best_individuals)
        else:
            context.population[-1] = deepcopy(context.best_individuals)