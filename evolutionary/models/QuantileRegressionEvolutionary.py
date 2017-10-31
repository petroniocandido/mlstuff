#!/usr/bin/python
# -*- coding: utf8 -*-

from mlstuff.evolutionary.EvolutionaryAlgorithm import Variable
from mlstuff.evolutionary.SimpleRealGA import SimpleRealGA
from mlstuff.evolutionary.EvolutionaryBoost import EvolutionaryBoost
from mlstuff.evolutionary import Util
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
    def __init__(self, tau, p, data,constant=True,**kwargs):
        super(QuantileRegressionEvolutionary, self)\
            .__init__(objectives_number = 1, variables_number = p+1 if constant else p,
                      best_individuals_size = 2,
                      crossover_rate=0.9, crossover_alpha_pol=0.9, crossover_alpha = 0.5,
                      mutation_rate=0.4, mutation_rand = lambda : np.random.uniform(-5,5),
                      selection_probability=0.7, selection_elitism_rate=0.3, **kwargs)

        self.p = p
        self.tau = tau
        self.data = data
        self.constant = constant

        if self.constant:
            self.variables.append(Variable(name="c", type=float, pmin=min(data),pmax=max(data)))
        for k in np.arange(0,self.p):
            self.variables.append(Variable(name="ar"+str(k), type=float, pmin=-5, pmax=5))

        self.objectives.append(Variable(name="y", type=int))

    def quantile_regression(self,individual, data):
        tmp = individual.variables['c'] if self.constant else 0
        for k in np.arange(0, self.p):
            tmp += individual.variables["ar"+str(k)] * data[k]
        individual.objectives['y'] = tmp

    def regularization_term(self,individual):
        tmp = abs(individual.variables['c']) if self.constant else 0
        for k in np.arange(0, self.p):
            tmp += abs(individual.variables["ar"+str(k)])
        return tmp ** 2

    def evaluate_individual(self, individual):

        forecasts = []

        for k in np.arange(self.p,len(self.data)):
            self.quantile_regression(individual, self.data[k-self.p: k])
            forecasts.append(individual.objectives['y'])

        individual.fitness = mean_pinball_loss(self.tau,self.data[self.p:],forecasts[:-1]) #+ self.regularization_term(individual)

    def stop_criteria(self):
        return self.best_fitness == 0


def getData():
    os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

    sunspotspd = pd.read_csv("DataSets/sunspots.csv", sep=",")
    sunspots = np.array(sunspotspd["SUNACTIVITY"][:])

    return sunspots

    #_min = min(sunspots)

    #_min = _min*0.9 if _min > 0 else _min*1.1

    #_max = max(sunspots)

    #_max = _max * 1.1 if _max > 0 else _max * 0.9

    #return  np.linspace(_min,_max,100).tolist()


def parallel_method_sample(**kwargs):
    sunspots = getData()

    l = int(len(sunspots)/2)

    instance = kwargs['instance']

    ini =  int(((instance % 3) * 0.5) * l)

    data = sunspots[ini : ini + l]

    time.sleep(0.6*instance)

    model = QuantileRegressionEvolutionary(0.7,2,data,constant= False,**kwargs)
    model.run()

    print('Instance: {0}    Fitness: {1}'.format(instance, model.best_individuals[0].fitness))

    return model.best_individuals


def parallel_method_full(**kwargs):
    sunspots = getData()

    l = int(len(sunspots)/2)

    instance = kwargs['instance']

    time.sleep(0.6*instance)

    model = QuantileRegressionEvolutionary(0.7,2,sunspots,constant= False,**kwargs)
    model.run()

    print('Instance: {0}    Fitness: {1}'.format(instance, model.best_individuals[0].fitness))

    return model.best_individuals


eb = EvolutionaryBoost(None,iterations=20,population_size=20,max_generations=250)

#best = eb.runParallelStochastic(parallel_method_sample, parallel_method_full)

best = eb.runParallel(parallel_method_full)

model = QuantileRegressionEvolutionary(0.7,2, getData(),constant= False)

#model.run()

Util.save_individuals(model, best, 'experiments/qr_sunspots_t07_p2_2.csv')

