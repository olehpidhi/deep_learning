from NeuralNet import *

import pandas as pd
data = pd.read_csv('breast-cancer.data', sep=",", header=None)
data.columns = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradit"]
for column in data:
    data[column], mapping_index = pd.factorize(data[column])

negative = data[data['class'] == 0]
positive = data[data['class'] == 1]

y_values = data['class']
data = data.drop(columns=['class'])

import numpy as np
from sklearn.model_selection import train_test_split

train = np.asarray(data.values, dtype = 'float32')
true_labels = np.asarray(y_values.values, dtype = 'float32')

batchsize = 50

X_train, X_test, y_train, y_test = train_test_split(train, true_labels, test_size=0.25, random_state=42)
X_train = X_train[0:50]
y_train = y_train[0:50]
X_test = X_test[0:50]
y_test = y_test[0:50]
mlp = MLP(layer_config=[9, 9, 9, 9], minibatch_size=50)
mlp.evaluate(np.array([X_train]), np.array([y_train]), np.array([X_test]), np.array([y_test]),
             eval_train=True)