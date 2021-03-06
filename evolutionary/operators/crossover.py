from numpy import random
import numpy as np

class SinglePointCrossover(object):
    def __init__(self, **kwargs):
        self.singlepointcrossover_enable = kwargs.get('singlepointcrossover_enable', lambda c: True)
        self.crossover_rate = kwargs.get('crossover_rate', 0.9)
        self.original_crossover_rate = self.crossover_rate
        self.crossover_alpha_pol = kwargs.get('crossover_alpha_pol', 0.9)
        self.crossover_alpha = kwargs.get('crossover_alpha', 0.5)
        self.crossover_dynamic_rate_increase = kwargs.get('crossover_dynamic_rate_increase', 0.01)

    def process(self, context):

        if not self.singlepointcrossover_enable(context):
            return

        if context.generations_without_improvement > 15:
            self.crossover_rate = min(self.crossover_rate + self.crossover_dynamic_rate_increase, 1)

        if context.generations_without_improvement == 0:
            self.crossover_rate = self.original_crossover_rate

        num_individuals = int(context.population_size * self.crossover_rate)

        for count in np.arange(0,num_individuals, step=2):
            direction = random.randint(0,2)
            crossover_point = random.randint(0, context.variables_number)

            tmp1 = random.randint(0, context.population_size)
            tmp2 = random.randint(0, context.population_size)

            while tmp1 == tmp2:
                tmp2 = random.randint(0, context.population_size)

            individual_best = context.population[tmp1] if context.population[tmp1].fitness <= context.population[tmp2].fitness \
                else context.population[tmp2]
            individual_worst = context.population[tmp2] if individual_best.fitness == context.population[tmp1].fitness \
                else context.population[tmp1]

            if direction == 1:
                for k in np.arange(crossover_point, context.variables_number):
                    var = context.variables[k]
                    best = individual_best.variables[var.name]
                    worst = individual_worst.variables[var.name]
                    new_best = self.crossover_alpha_pol * best + (1-self.crossover_alpha_pol) * worst
                    new_worst = self.crossover_alpha * best + (1 - self.crossover_alpha) * worst
                    individual_best.variables[var.name] = new_best
                    individual_worst.variables[var.name] = new_worst
            else:
                for k in np.arange(context.variables_number-1, crossover_point-1, step=-1):
                    var = context.variables[k]
                    best = individual_best.variables[var.name]
                    worst = individual_worst.variables[var.name]
                    new_best = self.crossover_alpha_pol * best + (1-self.crossover_alpha_pol) * worst
                    new_worst = self.crossover_alpha * best + (1 - self.crossover_alpha) * worst
                    individual_best.variables[var.name] = new_best
                    individual_worst.variables[var.name] = new_worst

            context.population[tmp1] = individual_best
            context.population[tmp2] = individual_worst
