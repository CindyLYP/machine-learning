import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1/(1+np.exp(-x))


def lr_train(features, labels, epoch=1000, learn_rate=0.000001):
    appd = np.ones((features.shape[0], 1))
    features = np.concatenate((features, appd), axis=1)
    w = np.array(np.random.rand(features.shape[1], 1))
    for index in range(epoch):
        index += 1
        loss = labels - sigmoid(np.dot(features, w))
        w += learn_rate*np.dot(features.T, loss)

        if index % 1 == 0:
            acc = evaluate(features, labels, w)
            print(' accuracy:', acc)
    return w


def predict(features, w):
    pred = sigmoid(np.dot(features, w))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred


def evaluate(features, labels, w, f=1):
    pred = predict(features, w)
    error = np.abs(labels - pred)
    acc = (labels.shape[0] - error.sum()) / labels.shape[0]
    return acc


if __name__ == "__main__":
    data_sets = load_breast_cancer()
    train_sets, test_sets, train_labels, test_labels = train_test_split(data_sets['data'], data_sets['target'],
                                                                        test_size=0.2, random_state=42)
    train_labels = train_labels.reshape((-1, 1))
    test_labels = test_labels.reshape((-1, 1))
    weight = lr_train(train_sets, train_labels, epoch=100)
    test_acc = evaluate(np.concatenate((test_sets, np.ones((test_sets.shape[0], 1))), axis=1), test_labels, weight, 2)
    print("test_acc:", test_acc)
