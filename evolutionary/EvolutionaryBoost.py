import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing


class EvolutionaryBoost(object):
    def __init__(self, model, iterations = 10, **kwargs):
        self.model = model
        self.iterations = iterations
        self.population_size = kwargs.get('population_size', 50)
        self.max_generations = kwargs.get('max_generations', 250)

    def run(self):
        best_individuals = []

        for i in np.arange(0,self.iterations):
            kwargs = {'population_size': self.population_size, 'max_generations': self.max_generations, 'dump': False}
            tmp = self.model(**kwargs)
            tmp.run()

            best_individuals.append( deepcopy(tmp.best_individuals) )

            print("Iteration: {0}   Fitness: {1}".format(i, tmp.best_individuals[0].fitness))

        best_individuals = np.ravel(best_individuals).tolist()

        best_individuals = sorted(best_individuals, key=lambda i: i.fitness)

        print("\nFinal evalutation\n")

        final_individuals = []

        for i in np.arange(0, self.iterations):
            kwargs = {'population_size':self.population_size, 'max_generations': self.max_generations, 'dump':False, \
                             'initial_population': deepcopy(best_individuals)}
            tmp = self.model(**kwargs)
            tmp.run()

            final_individuals.append(deepcopy(tmp.best_individuals))

            print("Iteration: {0}   Fitness: {1}".format(i, tmp.best_individuals[0].fitness))

        final_individuals = np.ravel(final_individuals).tolist()
        final_individuals = sorted(final_individuals, key=lambda i: i.fitness)

        return final_individuals

    def runParallel(self, parallel_method):

        num_cores = multiprocessing.cpu_count()

        best_individuals = Parallel(n_jobs=num_cores)(delayed(parallel_method)
                                                      (**{'population_size': self.population_size, 'max_generations': self.max_generations,
                                                          'dump': False, 'instance':i })
                                                      for i in np.arange(0,self.iterations))

        best_individuals = np.ravel(best_individuals).tolist()

        best_individuals = sorted(best_individuals, key=lambda i: i.fitness)

        print("\nFinal evalutation\n")

        final_individuals = []

        final_individuals = Parallel(n_jobs=num_cores)(delayed(parallel_method)
                                                      (**{'population_size': self.population_size, 'max_generations': self.max_generations,
                                                          'dump': False, 'initial_population': deepcopy(best_individuals), \
                                                          'instance': i})
                                                       for i in np.arange(0, self.iterations))

        final_individuals = np.ravel(final_individuals).tolist()
        final_individuals = sorted(final_individuals, key=lambda i: i.fitness)

        return final_individuals
