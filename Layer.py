import numpy as np

class Layer(object):
    def __init__(self, activation_function, inputs_count, outputs_count):
        self.activation_function = activation_function
        self.inputs_count = inputs_count
        self.outputs_count = outputs_count

        self.weights = np.random.uniform(-1.0, 1.0, size=self.outputs_count * (self.inputs_count + 1))
        self.weights = self.weights.reshape(self.outputs_count, self.inputs_count + 1)

        self.input = []
        self.weighed_input = []
        self.output = []


        print(self.weights.shape)

    def weigh(self, input):
        self.input = input
        self.weighed_input = self.weights.dot(input)
        return self.weighed_input

    def activate(self, input):
        self.output = self.activation_function(input)
        return self.output