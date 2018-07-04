from NeuralNet import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:

    def __init__(self, data, labels, batch_size, number_of_epochs):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.validation_loss = []
        self.train_loss = []
        self.validation_accuracies = []
        self.train_accuracies = []

    def data_split(self):
        train = self.data
        true_labels = self.labels
        X_train, X_test, y_train, y_test = train_test_split(train, true_labels, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    def indices_to_one_hot(self, data, nb_classes):
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def loss(self, predictions, y_hat):
        l = log_loss(y_hat, predictions)
        return l

    @staticmethod
    def batcherize(set, batch_size):
        for i in range(0, len(set), batch_size):
            yield set[i: min(i+batch_size, len(set) - 1)]

    def train_accuracy(self, predicted, true_value):
        predictions = predicted.argmax(axis=1)
        truth = np.argmax(true_value, axis=1)
        self.train_accuracies.append(sum(truth == predictions) / len(true_value))

    def accuracy(self, predicted, true_value):
        predictions = predicted.argmax(axis=1)
        truth = np.argmax(true_value, axis=1)
        self.validation_accuracies.append(sum(truth == predictions) / len(true_value))

    def fit(self, model):
        X_train, X_test, y_train, y_test = self.data_split()
        y_test_hat = self.indices_to_one_hot(y_test, 2)

        batch_size = 20

        batches = np.array(list(Trainer.batcherize(X_train, batch_size)))
        label_batches = np.array(list(Trainer.batcherize(y_train, batch_size)))

        for i in range(self.number_of_epochs):
            for b in range(len(batches)):
                output = model.forward(batches[b])
                y_hat = self.indices_to_one_hot(label_batches[b], 2)
                model.backward(output, y_hat, batches[b])

            output = model.forward(X_train)
            y_hat = self.indices_to_one_hot(y_train, 2)
            self.train_loss.append(self.loss(output, y_hat))
            self.train_accuracy(output, y_hat)

            predicted = model.forward(X_test)
            self.validation_loss.append(self.loss(predicted, y_test_hat))
            self.accuracy(predicted, y_test_hat)

        self.plot_model(self.train_loss, self.validation_loss, self.train_accuracies, self.validation_accuracies)
        # print(self.validation_accuracies)

    def plot_model(self, loss, val_loss, train_accuracies, validation_accuracy):
        # Create sub-plots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Summarize history for loss
        axs[0].plot(range(1, len(loss) + 1), loss)
        axs[0].plot(range(1, len(val_loss) + 1), val_loss)
        axs[0].set_title('Model Loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(val_loss) + 1), len(val_loss) / 10)
        axs[0].legend(['train', 'val'], loc='best')

        # Summarize history for accuracy
        axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies)
        axs[1].plot(range(1, len(validation_accuracy) + 1), validation_accuracy)
        axs[1].set_title('Model Accuracy')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(val_loss) + 1), len(val_loss) / 10)
        axs[1].legend(['train', 'val'], loc='best')

        # Show the plot
        plt.show()

    def predict(self, data, model):
        predictions = model.forward(data)
        return predictions

def get_data():
    data = pd.read_csv('breast-cancer.data', sep=",", header=None)
    data.columns = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast",
                    "breast-quad", "irradit"]
    for column in data:
        data[column], mapping_index = pd.factorize(data[column])

    labels = data['class']
    data = data.drop(columns=['class'])
    return data.values, labels.values

def build_model(learning_rate):
    num_of_features = 9
    dim_hidden_1 = 12
    dim_hidden_2 = 6
    dim_hidden_3 = 2
    layer = Layer(num_of_features, dim_hidden_1, ReLU)
    layer1 = Layer(dim_hidden_1, dim_hidden_2, tanh_f)
    layer2 = SkipLayer(dim_hidden_2, num_of_features)
    layer3 = Layer(dim_hidden_2, dim_hidden_3, softmax)

    return NeuralNet([layer, layer1, layer2, layer3], skip_layer=layer2, learning_rate=learning_rate)

def main():
    input, labels = get_data()
    model = build_model(learning_rate=0.001)
    trainer = Trainer(input, labels, 20, 100)
    trainer.fit(model=model)

main()




