#!/usr/bin/python
# -*- coding: utf8 -*-

from mlstuff.evolutionary.EvolutionaryAlgorithm import Variable
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA
import numpy as np
import os
import pandas as pd


def rmse(targets, forecasts):
    return np.sqrt(np.nanmean((targets - forecasts) ** 2))

class ARMAEvolutionary(SimpleRealGA):
    def __init__(self,p,q,data):
        super(ARMAEvolutionary, self)\
            .__init__(objectives_number = 1, variables_number = p+q+1, population_size = 50, max_generations = 50,
                      crossover_rate=0.9, crossover_alpha_pol=0.9, crossover_alpha = 0.5,
                      mutation_rate=0.4, mutation_rand = lambda : np.random.uniform(-5,5),
                      selection_probability=0.7, selection_elitism_rate=0.3)

        self.p = p
        self.q = q
        self.data = data

        self.variables.append(Variable(name="c", type=float, pmin=min(data),pmax=max(data)))
        for k in np.arange(0,self.p):
            self.variables.append(Variable(name="ar"+str(k), type=float, pmin=-10, pmax=10))
        for k in np.arange(0,self.q):
            self.variables.append(Variable(name="ma"+str(k), type=float, pmin=-10, pmax=10))

        self.objectives.append(Variable(name="y", type=int, pmin=min(data),pmax=max(data)))

    def arma(self,individual, data):
        tmp = individual.variables['c']
        for k in np.arange(0, self.p):
            tmp += individual.variables["ar"+str(k)] * data[k]
        individual.objectives['y'] = tmp

    def evaluate_individual(self, individual):

        forecasts = []

        for k in np.arange(self.p,len(self.data)):
            self.arma(individual, self.data[k-self.p: k])
            forecasts.append(individual.objectives['y'])

        individual.fitness = rmse(self.data[self.p:],forecasts)

    def stop_criteria(self):
        return self.best_fitness == 0

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

sunspotspd = pd.read_csv("DataSets/sunspots.csv", sep=",")
sunspots = np.array(sunspotspd["SUNACTIVITY"][:])

test = ARMAEvolutionary(3,0,sunspots)

test.run()

for i in test.best_individuals:
    print(i)
