import numpy as np
import pandas as pd
from Dataset import Dataset

from sklearn.model_selection import KFold

d = Dataset()
c_pos = d.get_c_pos()
pri_c0, pri_c1 = d.get_classes_prior()
cs = [0, 1]
dataset_layout = d.get_dataset_layout()


class LogReg():
    def __init__(self, X_train, y_train, n_steps, lr):
        self.X_train = X_train
        self.y_train = y_train
        self.n_steps = n_steps
        self.lr = lr
        self.weights = self.logistic_regression()

    def get_weights(self):
        return self.weights

    # Define the Logistic function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, weights):
        scores = np.dot(self.X_train, weights)
        return np.sum(self.y_train * scores - np.log(1 + np.exp(scores)))

    def logistic_regression(self):
        # weights = np.array([100.,0.,0.])
        weights = np.zeros(self.X_train.shape[1])

        for step in range(self.n_steps):
            mult = np.dot(self.X_train, weights)
            y_pred = self.sigmoid(mult)

            # Update weights with gradient
            error = self.y_train - y_pred
            gradient = np.dot(self.X_train.T, error)
            weights += self.lr * gradient

            # Print log-likelihood loss
            # if step % 1000 == 0:
            #     print("Loss: {}".format(self.log_likelihood(weights)))

        return weights

    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.weights)))

    def accuracy(self, X_test, y_test):
        n = X_test.shape[0]
        corrects = []
        for x, y in zip(X_test, y_test):
            y_pred = self.predict(x)
            corrects.append(y_pred - y)

        corrects = np.array(corrects)
        total_preds = np.count_nonzero(corrects == 0)
        return total_preds / n


# if __name__ == '__main__':
#     n_steps = 10000
#     lr = 0.001
#
#     X_train, X_test, y_train, y_test = d.get_train_test_sets()
#
#     X_train = np.array(
#         [[1, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, -1, 0]])
#     y_train = np.array([0, 0, 0, 1, 1, 1])
#
#     LR = LogReg(X_train, y_train, n_steps, lr)
#     acc = LR.accuracy(X_train, y_train)
#
#     print("[+]: The weight vector is: {}".format(LR.get_weights()))
#     print("The accuracy is: {}".format(acc))
