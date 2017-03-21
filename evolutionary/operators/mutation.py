from numpy import random


class GaussRandomMutation(object):
    def __index__(self, **kwargs):
        self.mutation_rate = kwargs.get('mutation_rate', 0.2)

    def process(self, context):
        num_individuals = int(context.population_size * self.mutation_rate)

        individuals = []

        for count in range(0, num_individuals):
            direction = random.randint(0, 1, 1)
            mutation_point = random.randint(0, context.variables_number, 1)

            tmp = random.randint(0, context.population_size, 1)

            while tmp in individuals:
                tmp = random.randint(0, context.population_size, 1)

            individual = context.population[tmp]

            if direction == 1:
                for k in range(mutation_point, context.variables_number):
                    var = context.variables[k]
                    individual.variables[var.name] += random.normal(0,1,1)
            else:
                for k in range(context.variables_number, mutation_point):
                    var = context.variables[k]
                    individual.variables[var.name] += random.normal(0, 1, 1)

            context.population[tmp] = individual
