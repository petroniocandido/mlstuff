#!/usr/bin/python
# -*- coding: utf8 -*-

from numpy import random
import numpy as np
from copy import deepcopy


class EvolutionaryAlgorithm(object):
    def __init__(self, **kwargs):
        self.objectives_number = kwargs.get('objectives_number',1)
        self.objectives = []
        self.variables_number = kwargs.get('variables_number', 1)
        self.variables = []
        self.population_size = kwargs.get('population_size', 50)
        self.max_generations = kwargs.get('max_generations', 500)
        self.operators = []
        self.population = []
        self.best_fitness = np.infty
        self.mean_fitness = 0
        self.generations_without_improvement = 0
        self.best_individuals_size = kwargs.get('best_individuals_size', 1)
        self.best_individuals = []
        self.dump = kwargs.get('dump', True)
        self.initial_population = kwargs.get('initial_population', None)

    def create_random_individual(self):
        variables = {}
        objectives = {}
        for var in self.variables:
            variables[var.name] = random.uniform(var.min, var.max)

        for var in self.objectives:
            objectives[var.name] = 0

        tmp = Individual(variables,objectives)

        self.check_individual_constraints(tmp)

        return tmp

    def create_initial_population(self):
        if self.initial_population is not None and self.generations_without_improvement == 0:
            self.population = self.initial_population
        else:
            self.population = [self.create_random_individual() for k in range(0, self.population_size) ]

    def check_individual_constraints(self, individual):
        for var in self.variables:
            if var.max is not None and individual.variables[var.name] > var.max:
                individual.variables[var.name] = var.max

            if var.min is not None and individual.variables[var.name] < var.min:
                individual.variables[var.name] = var.min

    def check_population_constraints(self):
        for individual in self.population:
            self.check_individual_constraints(individual)

    def evaluate_individual(self,individual):
        pass

    def evaluate_population(self):
        self.mean_fitness = 0
        for individual in self.population:
            self.evaluate_individual(individual)
            self.mean_fitness += individual.fitness

        self.mean_fitness /= self.population_size
        self.population = sorted(self.population, key=lambda i: i.fitness)

        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individuals = deepcopy(self.population[0:self.best_individuals_size])
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def stop_criteria(self):
        return False

    def run(self):
        generations = 0
        self.create_initial_population()
        while generations < self.max_generations and self.generations_without_improvement < 50 \
                and not self.stop_criteria():
            if self.dump:
                print('Generation: {0}   AVG Fitness: {1}   MIN Fitness: {2}'
                      .format(generations, self.mean_fitness, self.best_fitness))
            self.evaluate_population()
            for operator in self.operators:
                operator.process(self)

            self.check_population_constraints()

            generations += 1


class Variable(object):
    def __init__(self,name,type,pmin=None,pmax=None):
        self.name = name
        self.type = type
        self.min = pmin
        self.max = pmax


class Individual(object):
    def __init__(self,variables,objectives):
        self.fitness = 0
        self.variables = variables
        self.objectives = objectives

    def __str__(self):
        tmp = "Objectives:\n\t"
        for k in self.objectives.keys():
            tmp += '{0} = {1} \t'.format(k,self.objectives[k])

        tmp += "\nVariables:\n\t"
        for k in self.variables.keys():
            tmp += '{0} = {1} \t'.format(k,self.variables[k])

        tmp += '\nFitness: {0}'.format(self.fitness)

        return tmp
