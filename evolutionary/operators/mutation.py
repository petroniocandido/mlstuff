from numpy import random
import numpy as np
import time

class RealMutation(object):
    def __init__(self, **kwargs):
        self.mutation_rate = kwargs.get('mutation_rate', 0.3)
        self.original_mutation_rate = self.mutation_rate
        self.mutation_dynamic_rate_increase = kwargs.get('mutation_dynamic_rate_increase', 0.01)
        self.mutation_rand = kwargs.get('mutation_rand', lambda: random.normal(0, 1))
        t = time.localtime()
        random.seed(t.tm_year + t.tm_yday + t.tm_hour + t.tm_min + t.tm_sec)

    def process(self, context):

        if context.generations_without_improvement > 15:
            self.mutation_rate = min(self.mutation_rate + self.mutation_dynamic_rate_increase, 1)

        if context.generations_without_improvement == 0:
            self.mutation_rate = self.original_mutation_rate

        num_individuals = int(context.population_size * self.mutation_rate)

        individuals = []

        for count in np.arange(0, num_individuals):
            direction = random.randint(0, 1)
            mutation_point = random.randint(1, context.variables_number)

            tmp = random.randint(0, context.population_size)

            while tmp in individuals:
                tmp = random.randint(0, context.population_size)

            individual = context.population[tmp]

            if direction == 1:
                for k in np.arange(mutation_point, context.variables_number):
                    var = context.variables[k]
                    individual.variables[var.name] += self.mutation_rand()
            else:
                for k in np.arange(context.variables_number-1, mutation_point-1, step=-1):
                    var = context.variables[k]
                    individual.variables[var.name] += self.mutation_rand()

            context.population[tmp] = individual


class Reboot(object):
    def __init__(self, **kwargs):
        pass

    def process(self, context):
        if context.generations_without_improvement > 50:
            context.create_initial_population()