import numpy as np
from sklearn.neural_network._base import softmax, relu, tanh
from sklearn.metrics import log_loss

def ReLU(x, deriv=False):
    if deriv:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return relu(x)

def tanh_f(x, deriv=False):
    if deriv:
        return 1 - x * x
    else:
        return tanh(x)

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))


class Layer:

    def __init__(self, num_of_neurons, num_of_outputs, activationFunction):

        self.activationFunction = activationFunction
        self.num_of_neurons = num_of_neurons
        self.num_of_outputs = num_of_outputs
        self.xavier_init()

    def xavier_init(self):
        stdv = 2. / (self.num_of_neurons)
        self.W = np.random.uniform(-stdv, stdv, (self.num_of_neurons, self.num_of_outputs))
        self.B = np.random.uniform(-stdv, stdv, (1, self.num_of_outputs))

    def activate(self, input):
        self.input = input
        self.weighed_input = input.dot(self.W) + np.repeat(self.B, input.shape[0], axis=0)
        self.output = self.activationFunction(self.weighed_input)
        return self.output

    def error_signal(self, next_layer):
        derivative = self.activationFunction(self.weighed_input, True)
        self.delta = derivative * (next_layer.delta.dot(next_layer.W.T))

        return self.delta

    def update_weights(self, input, learning_rate):
        delta_weight = learning_rate * input.T.dot(self.delta)
        self.W -= delta_weight
        self.B -= np.sum(delta_weight, axis=0)


class SkipLayer:

    def __init__(self, num_of_neurons, num_of_outputs):
        self.Ws = np.eye(num_of_neurons, num_of_outputs)

    def activate(self, input):
        self.output = input + (self.skip_connection).dot(self.Ws.T)
        return self.output

    def update_skip_connection(self, skip_connection):
        self.skip_connection = skip_connection

    def update_weights(self, input, learning_rate):
        pass

    def error_signal(self, next_layer):
        self.delta = next_layer.delta
        self.W = next_layer.W
        return next_layer.delta


class NeuralNet:

    def __init__(self, layers, skip_layer, learning_rate):
        self.layers = layers
        self.skip_layer = skip_layer
        self.learning_rate = learning_rate
        self.loss = []

    def update_skip_connection(self, input):
        self.skip_layer.update_skip_connection(input)

    def forward(self, input):
        self.update_skip_connection(input)
        activation = self.layers[0].activate(input)
        for layer in self.layers[1:]:
            activation = layer.activate(activation)
        return activation

    def backward(self, input, y_hat, X):
        self.layers[-1].delta = (input - y_hat)
        self.layers[-2].error_signal(self.layers[-1])
        self.layers[-3].error_signal(self.layers[-2])
        self.layers[-4].error_signal(self.layers[-3])

        self.layers[-1].update_weights(self.layers[-2].output, self.learning_rate)
        self.layers[-2].update_weights(self.layers[-3].output, self.learning_rate)
        self.layers[-3].update_weights(self.layers[-4].output, self.learning_rate)
        self.layers[-4].update_weights(X, self.learning_rate)
