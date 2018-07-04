from NeuralNet import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class Trainer:

    def __init__(self, data, labels, learning_rates, batch_sizes, number_of_epochs):
        self.data = data
        self.labels = labels
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.number_of_epochs = number_of_epochs

    def data_split(self):
        train = self.data
        true_labels = self.labels
        X_train, X_test, y_train, y_test = train_test_split(train, true_labels, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    def indices_to_one_hot(self, data, nb_classes):
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def fit(self, model):
        X_train, X_test, y_train, y_test = self.data_split()

        for i in range(self.number_of_epochs):
            output = model.forward(X_train)
            y_hat = self.indices_to_one_hot(y_train, 2)
            model.backward(y_hat, output, X_train)

    def predict(self, data, model):
        predictions = model.forward(data)
        return predictions

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


# def indices_to_one_hot(data, nb_classes):
#     targets = np.array(data).reshape(-1)
#     return np.eye(nb_classes)[targets]


# y_values = indices_to_one_hot(labels.values, 2)
#
layer = Layer(num_of_features, dim_hidden_1, ReLU)
layer1 = Layer(num_of_features, dim_hidden_1, tanh_f)
layer2 = SkipLayer(dim_hidden_1, num_of_features, input)
layer3 = Layer(dim_hidden_1, dim_hidden_3, softmax)

network = NeuralNet([layer, layer1, layer2, layer3], skip_layer=layer2, learning_rate=0.01)
#
trainer = Trainer(data.values, labels.values, [0.1, 0.001, 0.0001], [10, 20, 50], 100)
trainer.fit(model=network)

# print(x.shape)
# print(y.shape)