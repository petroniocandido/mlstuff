import numpy as np


class UnivariateFunction(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name',None)

    def apply(self, data):
        if isinstance(data, (list, set, np.array)):
            return [self.function(k) for k in data]
        else:
            return self.function(data)

    def apply_derivative(self, data):
        if isinstance(data, (list, set, np.array)):
            return [ self.derivative(k) for k in data]
        else:
            return self.derivative(data)

    def function(self, data):
        pass

    def derivative(self, data):
        pass


class BivariateFunction(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name',None)

    def apply(self, x, y):
        if isinstance(x, (list, set, np.array)):
            l = len(x)
            return [self.function(x[k], y[k]) for k in range(l)]
        else:
            return self.function(x, y)

    def apply_derivative(self, x, y):
        if isinstance(x, (list, set, np.array)):
            l = len(x)
            return [self.derivative(x[k], y[k]) for k in range(l)]
        else:
            return self.derivative(x, y)

    def function(self, x, y):
        pass

    def derivative(self, x, y):
        pass