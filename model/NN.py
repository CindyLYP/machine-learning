import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1.0/(1+np.exp(-1*x))


def sig_derivative(x):
    return x*(1-x)


class NN:

    def __init__(self, layers=[], learning_rate=1e-6):  # 3,3,2 w [ 0,1 ]
        self.layers = layers
        self.w = list()
        self.b = list()
        self.lr = learning_rate
        for i in range(1, len(self.layers)):
            self.w.append((2*np.random.rand(self.layers[i], self.layers[i-1])-1)*0.25)
            self.b.append(np.random.rand(self.layers[i], 1))

    def fit(self, x, y, epochs=30):
        x = np.array(x)
        y = np.array(y)
        y = y.reshape(-1, 10, 1)
        for epoch in range(epochs):
            loss = 0
            for i in range(len(x)):
                # forward propagation
                a = [x[i].reshape(-1, 1)]
                for l in range(len(self.layers)-1):
                    a.append(sigmoid(np.dot(self.w[l], a[l])+self.b[l]))
                loss += np.sum(np.square(y[i]-a[-1])) / 2

                # back propagation
                delta = [-(y[i]-a[-1])*sig_derivative(a[-1])]
                i = 0
                for l in range(len(self.layers)-2, 0, -1):
                    delta.append(np.dot(self.w[l].T, delta[i])*sig_derivative(a[l]))
                    i += 1
                delta.reverse()

                # update weight,bias
                for l in range(len(self.w)):
                    self.w[l] -= self.lr*np.dot(delta[l], a[l].T)
                    self.b[l] -= self.lr*delta[l]
            print("epoch{}  loss: {}".format(epoch, loss/len(x)))

    def predict(self, x):
        y_pred = []
        for i in range(len(x)):
            a = x[i].reshape(-1, 1)
            for l in range(len(self.layers) - 1):
                tmp = np.dot(self.w[l], a)
                a = sigmoid(np.dot(self.w[l], a) + self.b[l])
            y_pred.append(a)
        y_pred = np.array(y_pred)
        return y_pred.reshape(y_pred.shape[0], -1)


if __name__ == "__main__":
    data = load_digits()
    X = data.data
    X -= X.min()
    X /= (X.max()-X.min())
    Y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
    y_train = LabelBinarizer().fit_transform(y_train)

    model = NN([x_train.shape[1], 8, 16, 10], learning_rate=0.1)
    model.fit(x_train, y_train, epochs=100)
    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=1)
    tmp = y_pred-y_test
    acc = np.sum(tmp==0)
    print("test accuracy: {}".format(acc/len(tmp)))