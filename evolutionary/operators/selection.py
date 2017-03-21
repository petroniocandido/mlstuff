from numpy import random


class RouletteWheel(object):
    def __index__(self,**kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)

    def process(self,context):
        num_individuals = int(context.population_size * self.selection_rate)

        individuals = []

        for count in range(0, num_individuals):
            pass


class Tournament(object):
    def __index__(self, **kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)
        self.selection_probability = kwargs.get('selection_probability', 0.7)

    def process(self, context):
        num_individuals = int(context.population_size * self.selection_rate)

        individuals = []

        for count in range(0, num_individuals):
            tmp1 = random.randint(0, context.population_size, 1)
            tmp2 = random.randint(0, context.population_size, 1)

            while tmp1 == tmp2:
                tmp2 = random.randint(0, context.population_size, 1)

            individual_best = tmp1 if context.population[tmp1].fitness <= context.population[tmp2].fitness else tmp2
            individual_worst = tmp2 if individual_best == tmp1 else tmp1

            p = random.uniform(0,1,1)

            if p <= self.selection_probability:
                individuals.append(individual_best)
            else:
                individuals.append(individual_worst)

        context.population = individuals