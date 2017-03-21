from numpy import random


class SinglePointCrossover(object):
    def __index__(self, **kwargs):
        self.crossover_rate = kwargs.get('crossover_rate', 0.9)
        self.crossover_alpha_pol = kwargs.get('crossover_alpha_pol', 0.9)
        self.crossover_alpha = kwargs.get('crossover_alpha', 0.5)

    def process(self, context):
        num_individuals = int(context.population_size * self.crossover_rate)

        for count in range(0,num_individuals,step=2):
            direction = random.randint(0,1,1)
            crossover_point = random.randint(0, context.variables_number, 1)

            tmp1 = random.randint(0, context.population_size, 1)
            tmp2 = random.randint(0, context.population_size, 1)

            while tmp1 == tmp2:
                tmp2 = random.randint(0, context.population_size, 1)

            indivualBest = tmp1 if context.population[tmp1].fitness <= context.population[tmp2].fitness else tmp2
            indivualWorst = tmp2 if indivualBest == tmp1 else tmp1

            if direction == 1:
                for k in range(crossover_point, context.variables_number):
                    var = context.variables[k]
                    best = indivualBest.variables[var.name]
                    worst = indivualWorst.variables[var.name]
                    new_best = self.crossover_alpha_pol * best + (1-self.crossover_alpha_pol) * worst
                    new_worst = self.crossover_alpha * best + (1 - self.crossover_alpha) * worst
                    indivualBest.variables[var.name] = new_best
                    indivualWorst.variables[var.name] = new_worst
            else:
                for k in range(context.variables_number, crossover_point):
                    var = context.variables[k]
                    best = indivualBest.variables[var.name]
                    worst = indivualWorst.variables[var.name]
                    new_best = self.crossover_alpha_pol * best + (1-self.crossover_alpha_pol) * worst
                    new_worst = self.crossover_alpha * best + (1 - self.crossover_alpha) * worst
                    indivualBest.variables[var.name] = new_best
                    indivualWorst.variables[var.name] = new_worst

            context.population[tmp1] = indivualBest
            context.population[tmp2] = indivualWorst