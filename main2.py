import pandas as pd
data = pd.read_csv('breast-cancer.data.txt', sep=",", header=None)
data.columns = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradit"]
for column in data:
    data[column], mapping_index = pd.factorize(data[column])

negative = data[data['class'] == 0]
positive = data[data['class'] == 1]

y_values = data['class']
data = data.drop(columns = ['class'])

nn = NeuralNet(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=50,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)
