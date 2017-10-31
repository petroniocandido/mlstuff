#!/usr/bin/python
# -*- coding: utf8 -*-

from mlstuff.evolutionary.EvolutionaryAlgorithm import Variable
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA
from mlstuff.evolutionary.EvolutionaryBoost import EvolutionaryBoost
import numpy as np
import os
import pandas as pd
import numpy.random as random
import time
from mlstuff.evolutionary import Util


def rmse(targets, forecasts):
    return np.sqrt(np.nanmean((targets - forecasts) ** 2))


def residual_mean(targets, forecasts):
    return np.nanmean(targets - forecasts)


class ARMAEvolutionary(SimpleRealGA):
    def __init__(self,p,q,data,constant= False,**kwargs):
        super(ARMAEvolutionary, self)\
            .__init__(objectives_number = 1, variables_number = p+q+1 if constant else  p+ q, best_individuals_size = 2,
                      crossover_rate=0.9, crossover_alpha_pol=0.9, crossover_alpha = 0.5,
                      mutation_rate=0.4, mutation_rand = lambda : np.random.uniform(-5,5),
                      selection_probability=0.7, selection_elitism_rate=0.3, **kwargs)

        self.p = p
        self.q = q
        self.data = data
        self.constant= constant

        if self.constant:
            self.variables.append(Variable(name="c", type=float, pmin=min(data),pmax=max(data)))
        for k in np.arange(0,self.p):
            self.variables.append(Variable(name="ar"+str(k), type=float, pmin=-5, pmax=5))
        for k in np.arange(0,self.q):
            self.variables.append(Variable(name="ma"+str(k), type=float, pmin=-10, pmax=10))

        self.objectives.append(Variable(name="y", type=int, pmin=min(data),pmax=max(data)))

    def arma(self,individual, data):
        tmp = individual.variables['c'] if self.constant else 0
        for k in np.arange(0, self.p):
            tmp += individual.variables["ar"+str(k)] * data[k]
        individual.objectives['y'] = tmp

    def regularization_term(self,individual):
        tmp = abs(individual.variables['c']) if self.constant else 0
        for k in np.arange(0, self.p):
            tmp += abs(individual.variables["ar"+str(k)])
        return  tmp ** 2

    def evaluate_individual(self, individual):

        forecasts = []

        for k in np.arange(self.p,len(self.data)):
            self.arma(individual, self.data[k-self.p: k])
            forecasts.append(individual.objectives['y'])

        individual.fitness = rmse(self.data[self.p:],forecasts) \
                             + (residual_mean(self.data[self.p:],forecasts) ** 2) + self.regularization_term(individual)

    def stop_criteria(self):
        return self.best_fitness == 0

def getData():
    os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

    sunspotspd = pd.read_csv("DataSets/sunspots.csv", sep=",")
    sunspots = np.array(sunspotspd["SUNACTIVITY"][:])
    return  sunspots

def parallel_method(**kwargs):
    sunspots = getData()

    instance = kwargs['instance']

    time.sleep(0.5*instance)

    model = ARMAEvolutionary(2,0,sunspots,constant= False,**kwargs)
    model.run()

    print('Instance: {0}    Fitness: {1}'.format(instance, model.best_individuals[0].fitness))

    return model.best_individuals

eb = EvolutionaryBoost(None,iterations=20,population_size=10,max_generations=50)

best = eb.runParallel(parallel_method)

model = ARMAEvolutionary(2,0,getData(),constant= False)

Util.save_individuals(model, best, 'experiments/arma_sunspots_p2_q0.csv')

#for i in best:
#    print(i)

