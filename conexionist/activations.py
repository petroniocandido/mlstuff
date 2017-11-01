import numpy as np
import mlstuff.conexionist.function as func


class Identity(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(BinaryStep, self).__init__(name='Binary Step', **kwargs)

    def function(self, data):
        return data

    def derivative(self, data):
        return 1


class BinaryStep(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(BinaryStep, self).__init__(name='Binary Step', **kwargs)

    def function(self, data):
        return 1 if data >= 0 else 0

    def derivative(self, data):
        return 0 if data >= 0 else 0


class Linear(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(BinaryStep, self).__init__(name='Linear', **kwargs)
        self.a = kwargs.get('coefficient', 1)
        self.b = kwargs.get('threshold', 0)

    def function(self, data):
        return self.a*data + self.b

    def derivative(self, data):
        return self.a


class Sigmoid(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(name='Sigmoid', **kwargs)

    def function(self, data):
        return 1/(1 + np.exp(-data))

    def derivative(self, data):
        k = self.function(data)
        return  k * (1 - k)


class TanH(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(TanH, self).__init__(name='Hyperbolic Tangent', **kwargs)

    def function(self, data):
        return 2/(1 + np.exp(-2 * data)) - 1

    def derivative(self, data):
        k = self.function(data)
        return  1 - k**2


class ReLu(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(ReLu, self).__init__(name='Rectified linear unit', **kwargs)

    def function(self, data):
        return np.max(0, data)

    def derivative(self, data):
        return  1 if data >= 0 else 0


class LeakyReLu(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(LeakyReLu, self).__init__(name='Leaky Rectified linear unit', **kwargs)
        self.a = kwargs.get('a', .01)

    def function(self, data):
        return self.a*data if data < 0 else data

    def derivative(self, data):
        return self.a if data < 0 else 1


class Softmax(func.UnivariateFunction):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(name='Softmax', **kwargs)

    def function(self, data):
        tmp = [np.exp(k) for k in data]
        Z = np.sum(tmp)
        return [k/Z for k in tmp]

    def derivative(self, data):
        ret = []
        for i,pi in enumerate(data):
            tmp = []
            for j, pj in enumerate(data):
                if i == j:
                    tmp.append(self.function(pi)*(1-self.function(pj)))
                else:
                    tmp.append(- self.function(pi) * self.function(pj))
            ret.append(tmp)
        return ret