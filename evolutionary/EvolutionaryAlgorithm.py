#!/usr/bin/python
# -*- coding: utf8 -*-

from numpy import random
import numpy as np

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

    def create_random_individual(self):
        variables = {}
        objectives = {}
        for var in self.variables:
            variables[var.name] = random.uniform(var.min, var.max, 1)

        for var in self.objectives:
            objectives[var.name] = 0

        return Individual(variables,objectives)

    def create_initial_population(self):
        self.population = [self.create_random_individual() for k in range(0, self.population_size) ]

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
            self.best_individuals = self.population[0:self.best_individuals_size]
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def stop_criteria(self):
        return False

    def run(self):
        generations = 0
        self.create_initial_population()
        while generations < self.max_generations  and not self.stop_criteria():
            if self.dump:
                print('Generation: {0}   AVG Fitness: {1}   MIN Fitness: {2}'.format(generations, self.mean_fitness, self.best_fitness))
            self.evaluate_population()
            for operator in self.operators:
                operator.process(self)
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
            tmp += k + "=" + str(self.objectives[k]) + "\t"

        tmp += "Variables:\n\t"
        for k in self.variables.keys():
            tmp += k + "=" + str(self.variables[k]) +"\t"

        tmp += "\nFitness: " + str(self.fitness)

        return tmp
