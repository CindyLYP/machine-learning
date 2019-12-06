import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegression:
    def __init__(self, alpha, shape):
        self.a = alpha
        self.m = shape[0]
        self.w = (2*np.random.rand(shape[1], 1)-1)*0.25

    def fit(self, data, labels, epochs=30):
        print(labels.shape)
        for epoch in range(epochs):
            tmp = np.dot(data, self.w)-labels
            partical = np.dot(tmp.T, data)/self.m
            self.w = self.w-self.a*partical.T
            if epoch % 100 == 0:
                loss = np.sum(np.square(tmp)) / self.m
                print("loss: {:.6f}".format(loss))

    def predict(self, data):
        return np.dot(data, self.w)


def preprocess(train_x, test_x, train_y, test_y):
    train_sets = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    test_sets = np.concatenate((test_x, np.ones((test_x.shape[0], 1))), axis=1)
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    data_sets, label_sets = load_boston(return_X_y=True)
    train_sets, test_sets, train_labels, test_labels = \
        train_test_split(data_sets, label_sets, test_size=0.2, random_state=42)
    train_sets, test_sets, train_labels, test_labels = preprocess(train_sets, test_sets, train_labels, test_labels)
    # print(train_sets, train_labels)
    alpha = 1e-7
    print(alpha)
    model = LinearRegression(alpha, train_sets.shape)
    model.fit(train_sets, train_labels, epochs=30000)
    pred = model.predict(test_sets)
    pred = np.around(pred, decimals=1)
    print("test loss: {:.6f}".format(np.sum(np.square(pred-test_labels)) / test_labels.shape[0]))
    print(test_labels[:20].T, '\n', pred[:20].T)