import numpy as np
from mlstuff.conexionist import costs, activations


class Layer(object):

    def __init__(self, num_inputs, num_neurons, activation=activations.Identity, **kwargs):
        self.name = kwargs.get("name", None)
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = np.random.normal(scale=0.1, size=(self.num_inputs, self.num_neurons))
        self.bias = np.zeros(self.num_neurons)
        self.input = None
        self.output = None

    def propagate(self, input, **kwargs):
        '''
        Output = Activation( Input * Weights + Bias )
        :param input:
        :param kwargs:
        :return:
        '''
        self.input = input
        self.output = self.activation.apply( self.input.dot(self.weights) + self.bias )
        return self.output

    def back_propagate(self, delta, eta, **kwargs):
        '''
        dOutput/dWeight = dOutput/dActivation * dActivation/dWeight
        dOutput/dBias = dOutput/dActivation * dActivation/dBias

        Weight = Weight - eta * dOutput/dWeight
        Bias = Bias - eta * dOutput/dBias
        :param delta:
        :param eta:
        :param kwargs:
        :return:
        '''
        out_delta = delta * self.activation.apply_derivative(self.output)
        tmp = out_delta.dot(self.weights.T)
        self.weights -= self.input.dot(tmp.T) * eta
        return tmp

    def __str__(self):
        tmp = str(self.name) if self.name is not None else "Layer"
        tmp += "(" + str(self.num_inputs) + "," + str(self.num_neurons) + ")"
        return tmp


class Network(object):
    def __init__(self, cost=costs.Quadratic, **kwargs):
        self.num_inputs = None
        self.num_outputs = None
        self.cost = cost
        self.layers = []
        self.transformations = []

    def append_layer(self,layer):
        if len(self.layers) == 0:
            self.num_inputs = layer.num_inputs
        self.num_outputs = layer.num_neurons

        self.layers.append(layer)

    def propagate(self, input, **kwargs):
        tmp_in = input
        for layer in self.layers:
            tmp_in = layer.propagate(tmp_in, **kwargs)
        return tmp_in

    def back_propagate(self, delta, eta, **kwargs):
        tmp_out = delta
        for layer in reversed(self.layers):
            tmp_out = layer.back_propagate(tmp_out, eta, **kwargs)
        return tmp_out

    def train(self, x, y, **kwargs):
        steps = kwargs.get("num_iterations", 100)
        batch_size = kwargs.get("batch_size", 1)
        eta = kwargs.get("learning_rate", 0.1)
        error = []
        for i in range(steps):
            for batch in range(int(len(x) / batch_size)):
                tmp_x = x[batch * batch_size: (batch+1)*batch_size]
                tmp_y = y[batch * batch_size: (batch + 1) * batch_size]

                out = self.propagate(tmp_x, **kwargs)
                delta = tmp_y - out

                self.back_propagate(delta, eta, **kwargs)

                error.append(delta)

        return error

