import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

c_pos = 0
cs = [0, 1]
n_classes = 2
dataset_layout = ["attr1", "attr2", "attr3", "attr4", "class"]

for i, name in enumerate(dataset_layout):
    if name == "class":
        c_pos = i

# ================== Gathering the data ================== #
def get_train_test_sets(kfold):
    data_fname = 'data_banknote_authentication.txt'
    df = pd.read_csv(data_fname, sep=',', header=None)
    data = df.values

    kfold = KFold(kfold, True, 0)
    X = data[:, 0:c_pos]
    y = data[:, c_pos]

    for train_index, test_index in kfold.split(data):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test
# ================== ================== ================== #


class LogReg():
    def __init__(self, X_train, y_train, n_steps, lr):
        self.X_train = X_train
        self.y_train = y_train
        self.n_steps = n_steps
        self.lr = lr
        self.weights = self.logistic_regression()

    # - Define the Logistic function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_likelihood(self, weights):
        scores = np.dot(self.X_train, weights)
        return np.sum(self.y_train*scores - np.log(1 + np.exp(scores)))

    def logistic_regression(self):
        weights = np.zeros(self.X_train.shape[1])

        for step in range(self.n_steps):
            scores = np.dot(self.X_train, weights)
            predictions = self.sigmoid(scores)

            # Update weights with gradient
            output_error_signal = self.y_train - predictions
            gradient = np.dot(self.X_train.T, output_error_signal)
            weights += self.lr * gradient

            # Print log-likelihood loss
            if step % 1000 == 0:
                print("Loss: {}".format(self.log_likelihood(weights)))

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
        return total_preds/n


if __name__ == '__main__':
    kfold = 3
    n_steps = 10000
    lr = 0.001
    X_train, X_test, y_train, y_test = get_train_test_sets(kfold)

    LR = LogReg(X_train, y_train, n_steps, lr)
    acc = LR.accuracy(X_test, y_test)
    print()
    print("The accuracy is: {}".format(acc))
