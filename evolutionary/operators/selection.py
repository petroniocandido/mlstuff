from numpy import random


class RouletteWheel(object):
    def __index__(self,**kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)

    def process(self,population):
        pass


class Tournament(object):
    def __index__(self, **kwargs):
        self.selection_rate = kwargs.get('selection_rate', 1)

    def process(self, population):
        pass