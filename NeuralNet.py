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


# num_of_features = 9
# dim_hidden_1 = 9
# dim_hidden_3 = 2
# input = np.ones((15, 9))
#
# layer = Layer(num_of_features, dim_hidden_1, ReLU)
# layer1 = Layer(num_of_features, dim_hidden_1, tanh_f)
# layer2 = SkipLayer(dim_hidden_1, num_of_features, input)
# layer3 = Layer(dim_hidden_1, dim_hidden_3, softmax)
#
# network = NeuralNet([layer, layer1, layer2, layer3])
# output = network.forward(input)
# print(output)
# network.backward(output, np.eye(15, dim_hidden_3), input)
# output = network.forward(input)
# print(output)

#
# class Layer:
#     def __init__(self, size, minibatch_size, is_input=False, is_output=False,
#                  activation=f_sigmoid):
#         self.is_input = is_input
#         self.is_output = is_output
#
#         # Z is the matrix that holds output values
#         self.Z = np.zeros((minibatch_size, size[0]))
#         # The activation function is an externally defined function (with a
#         # derivative) that is stored here
#         self.activation = activation
#
#         # W is the outgoing weight matrix for this layer
#         self.W = None
#         # S is the matrix that holds the inputs to this layer
#         self.S = None
#         # D is the matrix that holds the deltas for this layer
#         self.D = None
#         # Fp is the matrix that holds the derivatives of the activation function
#         self.Fp = None
#
#         if not is_input:
#             self.S = np.zeros((minibatch_size, size[0]))
#             self.D = np.zeros((minibatch_size, size[0]))
#
#         if not is_output:
#             self.W = np.random.normal(size=size, scale=1E-4)
#
#         if not is_input and not is_output:
#             self.Fp = np.zeros((size[0], minibatch_size))
#
#     def forward_propagate(self):
#         if self.is_input:
#             return self.Z.dot(self.W)
#
#         self.Z = self.activation(self.S)
#         if self.is_output:
#             return self.Z
#         else:
#             # For hidden layers, we add the bias values here
#             self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
#             self.Fp = self.activation(self.S, deriv=True).T
#             return self.Z.dot(self.W)

#
# class MLP:
#
#     def __init__(self, layer_config, minibatch_size=100):
#         self.layers = []
#         self.num_layers = len(layer_config)
#         self.minibatch_size = minibatch_size
#
#         for i in range(self.num_layers-1):
#             if i == 0:
#                 print("Initializing input layer with size {0}.".format(
#                     layer_config[i]
#                 ))
#                 # Here, we add an additional unit at the input for the bias
#                 # weight.
#                 self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
#                                          minibatch_size,
#                                          is_input=True))
#             else:
#                 print("Initializing hidden layer with size {0}.".format(
#                     layer_config[i]
#                 ))
#                 # Here we add an additional unit in the hidden layers for the
#                 # bias weight.
#                 self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
#                                          minibatch_size,
#                                          activation=f_sigmoid))
#
#         print("Initializing output layer with size {0}.".format(
#             layer_config[-1]
#         ))
#         self.layers.append(Layer([layer_config[-1], None],
#                                  minibatch_size,
#                                  is_output=True,
#                                  activation=f_softmax))
#         print("Done!")
#
#     def forward_propagate(self, data):
#         # We need to be sure to add bias values to the input
#         self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)
#
#         for i in range(self.num_layers-1):
#             self.layers[i+1].S = self.layers[i].forward_propagate()
#         return self.layers[-1].forward_propagate()
#
#     def backpropagate(self, yhat, labels):
#         self.layers[-1].D = (yhat - labels).T
#         for i in range(self.num_layers-2, 0, -1):
#             # We do not calculate deltas for the bias values
#             W_nobias = self.layers[i].W[0:-1, :]
#
#             self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * \
#                                self.layers[i].Fp
#
#     def update_weights(self, eta):
#         for i in range(0, self.num_layers-1):
#             W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
#             self.layers[i].W += W_grad
#
#
#     def evaluate(self, train_data, train_labels, test_data, test_labels,
#                  num_epochs=500, eta=0.05, eval_train=False, eval_test=True):
#
#         N_train = len(train_labels)*len(train_labels[0])
#         N_test = len(test_labels)*len(test_labels[0])
#
#         print("Training for {0} epochs...".format(num_epochs))
#         for t in range(0, num_epochs):
#             out_str = "[{0:4d}] ".format(t)
#
#             for b_data, b_labels in zip(train_data, train_labels):
#                 output = self.forward_propagate(b_data)
#                 self.backpropagate(output, b_labels)
#                 self.update_weights(eta=eta)
#
#             if eval_train:
#                 errs = 0
#                 for b_data, b_labels in zip(train_data, train_labels):
#                     output = self.forward_propagate(b_data)
#                     yhat = np.argmax(output, axis=1)
#                     errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])
#
#                 out_str = "{0} Training error: {1:.5f}".format(out_str,
#                                                            float(errs)/N_train)
#
#             if eval_test:
#                 errs = 0
#                 for b_data, b_labels in zip(test_data, test_labels):
#                     output = self.forward_propagate(b_data)
#                     yhat = np.argmax(output, axis=1)
#                     errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])
#
#                 out_str = "{0} Test error: {1:.5f}".format(out_str,
#                                                        float(errs)/N_test)
#
#             print(out_str)
