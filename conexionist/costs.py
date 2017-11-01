import numpy as np
import mlstuff.conexionist.function as func


class Quadratic(func.BivariateFunction):
    def __init__(self, **kwargs):
        super(Quadratic, self).__init__(name='Quadratic Cost Function', **kwargs)

    def apply(self, x, y):
        n = len(x)
        ret = 0
        for i in range(n):
            ret += (x[i] - y[i])**2
        return ret/n


    def derivative(self, x, y):
        pass


class CrossEntropy(func.BivariateFunction):
    def __init__(self, **kwargs):
        super(Quadratic, self).__init__(name='Quadratic Cost Function', **kwargs)

    def apply(self, x, y):
        n = len(x)
        ret = 0
        for i in range(n):
            ret += y[i]*np.log(x[i]) + (1 - y[i])*np.log(1 - x[i])
        return -ret/n

    def derivative(self, x, y):
        pass