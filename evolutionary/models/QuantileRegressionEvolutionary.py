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


def pinball_loss(tau, data, u):
    qy = (tau - 1) * sum([ii - u for ii in data if ii < u]) + tau * sum([ii - u for ii in data if ii >= u])
    return qy


def mean_pinball_loss(tau, data, forecasts):
    qy = np.mean([pinball_loss(tau, data, u) for u in forecasts ])
    return qy

class QuantileRegressionEvolutionary(SimpleRealGA):
    def __init__(self, tau, p, data,**kwargs):
        super(QuantileRegressionEvolutionary, self)\
            .__init__(objectives_number = 1, variables_number = p+1, best_individuals_size = 2,
                      crossover_rate=0.9, crossover_alpha_pol=0.9, crossover_alpha = 0.5,
                      mutation_rate=0.4, mutation_rand = lambda : np.random.uniform(-5,5),
                      selection_probability=0.7, selection_elitism_rate=0.3, **kwargs)

        self.p = p
        self.tau = tau
        self.data = data

        self.variables.append(Variable(name="c", type=float, pmin=min(data),pmax=max(data)))
        for k in np.arange(0,self.p):
            self.variables.append(Variable(name="ar"+str(k), type=float, pmin=-100, pmax=100))

        self.objectives.append(Variable(name="y", type=int))

    def quantile_regression(self,individual, data):
        tmp = individual.variables['c']
        for k in np.arange(0, self.p):
            tmp += individual.variables["ar"+str(k)] * data[k]
        individual.objectives['y'] = tmp

    def evaluate_individual(self, individual):

        forecasts = []

        for k in np.arange(self.p,len(self.data)):
            self.quantile_regression(individual, self.data[k-self.p: k])
            forecasts.append(individual.objectives['y'])

        individual.fitness = mean_pinball_loss(self.tau,self.data,forecasts)

    def stop_criteria(self):
        return self.best_fitness == 0

def parallel_method(**kwargs):
    os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

    sunspotspd = pd.read_csv("DataSets/sunspots.csv", sep=",")
    sunspots = np.array(sunspotspd["SUNACTIVITY"][:])

    instance = kwargs['instance']

    time.sleep(0.5*instance)

    model = QuantileRegressionEvolutionary(0.5,2,sunspots,**kwargs)
    model.run()

    print('Instance: {0}    Fitness: {1}'.format(instance, model.best_individuals[0].fitness))

    return model.best_individuals

#os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

#sunspotspd = pd.read_csv("DataSets/sunspots.csv", sep=",")
#sunspots = np.array(sunspotspd["SUNACTIVITY"][:])

#model = lambda **k: ARMAEvolutionary(2,0,sunspots,**k)

eb = EvolutionaryBoost(None,iterations=20,population_size=20,max_generations=250)

best = eb.runParallel(parallel_method)

for i in best:
    print(i)

#test = ARMAEvolutionary(2,0,sunspots)

#test.run()

#for i in test.best_individuals:
#    print(i)
