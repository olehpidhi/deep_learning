from NeuralNet import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

class Trainer:

    def __init__(self, data, labels, learning_rates, batch_sizes, number_of_epochs):
        self.data = data
        self.labels = labels
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.number_of_epochs = number_of_epochs

    def data_split(self):
        train = np.asarray(self.data, dtype='float32')
        true_labels = np.asarray(self.labels, dtype='float32')
        X_train, X_test, y_train, y_test = train_test_split(train, true_labels, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def indices_to_one_hot(data, nb_classes):
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def train(self, model):
        X_train, X_test, y_train, y_test = self.data_split()

        for i in range(self.number_of_epochs):
            model.forward(X_train)
            model.backward(output, y_values, input)

        return model.forward(self.data)

import pandas as pd

data = pd.read_csv('breast-cancer.data', sep=",", header=None)
data.columns = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradit"]
for column in data:
    data[column], mapping_index = pd.factorize(data[column])

negative = data[data['class'] == 0]
positive = data[data['class'] == 1]

labels = data['class']
data = data.drop(columns=['class'])

num_of_features = 9
dim_hidden_1 = 9
dim_hidden_3 = 2
input = data.values


y_values = Trainer.indices_to_one_hot(labels.values, 2)

layer = Layer(num_of_features, dim_hidden_1, ReLU)
layer1 = Layer(num_of_features, dim_hidden_1, tanh_f)
layer2 = SkipLayer(dim_hidden_1, num_of_features, input)
layer3 = Layer(dim_hidden_1, dim_hidden_3, softmax)

network = NeuralNet([layer, layer1, layer2, layer3], 0.1)

output = network.forward(input)
network.backward(output, y_values, input)
output = network.forward(input)
network.backward(output, y_values, input)
output = network.forward(input)
print(network.loss)

# trainer = Trainer(np.array(data.values), np.array(labels.values), [0.1, 0.001, 0.0001], [10, 20, 50], 100)
# trainer.random_batch(data.values, labels.values, 30)


# print(x.shape)
# print(y.shape)