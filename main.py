from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from NeuralNet import *
from Layer import *

mndata = MNIST('../python-mnist/data')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

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


def sigmoid(z, derivative=False):
    if derivative:
        sg = expit(z)
        return sg * (1.0 - sg)
    else:
        return expit(z)


nn.add_layer(Layer(sigmoid, X_train.shape[1], 50))
nn.add_layer(Layer(sigmoid, 50, 10))
# nn.add_layer(Layer())
nn.fit(X_train, Y_train, print_progress=True)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
# plt.savefig('./figures/cost.png', dpi=300)
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
#plt.savefig('./figures/cost2.png', dpi=300)
plt.show()

y_train_pred = nn.predict(X_train)

acc = np.sum(Y_train == y_train_pred, axis=0) / X_train.shape[0]

print('Training accuracy: %.2f%%' % (acc * 100))

Y_test_pred = nn.predict(X_test)

acc = np.sum(Y_test == Y_test_pred, axis=0) / X_test.shape[0]

print('Test accuracy: %.2f%%' % (acc * 100))

miscl_img = X_test[Y_test != Y_test_pred][:25]
correct_lab = Y_test[Y_test != Y_test_pred][:25]
miscl_lab = Y_test_pred[Y_test != Y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('./figures/mnist_miscl.png', dpi=300)
plt.show()
