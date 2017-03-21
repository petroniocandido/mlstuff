

class EvolutionaryAlgorithm(object):
    def __init__(self, **kwargs):
        self.objectives_number = kwargs.get('objectives_number',1)
        self.objectives = []
        self.variables_number = kwargs.get('variables_number', 1)
        self.variables = []
        self.population_size = kwargs.get('population_size', 1)
        self.max_generations = kwargs.get('max_generations', 1)
        self.operators = []
        self.population = []
        self.best_fitness = 0
        self.mean_fitness = 0

    def create_random_individual(self):
        pass

    def create_initial_population(self):
        self.population = [self.create_random_individual() for k in range(0, self.population_size) ]

    def evaluate_individual(self,individual):
        pass

    def evaluate_population(self):
        for individual in self.population:
            self.evaluate_individual(individual)

    def stop_criteria(self):
        return False

    def run(self):
        generations = 0
        self.create_initial_population()
        while generations < self.max_generations and not self.stop_criteria():
            self.evaluate_population()
            for operator in self.operators:
                operator.process(self.population)
            generations += 1


class Variable(object):
    def __init__(self,name,type,pmin=None,pmax=None):
        self.name = name
        self.type = type
        self.min = pmin
        self.max = pmax


class Individual(object):
    def __init__(self):
        self.fitness = 0