import numpy as np
import pandas as pd

c_pos = 0
cs = [0, 1]
n_classes = 2
dataset_layout = ["attr1", "attr2", "attr3", "attr4", "class"]

for i, name in enumerate(dataset_layout):
    if name == "class":
        c_pos = i

# ================== Gathering the data ================== #
data_fname = 'data_banknote_authentication.txt'
df = pd.read_csv(data_fname, sep=',', header=None)
data = df.values
X = df.iloc[:, 0:c_pos].values
y = df[c_pos].values
n_total_rows = X.shape[0]

# Get all data values for Class 0 and Class 1
c0_X = df.loc[df[c_pos] == cs[0]].iloc[:, 0:c_pos].values
c1_X = df.loc[df[c_pos] == cs[1]].iloc[:, 0:c_pos].values
# ================== ================== ================== #

# - Define the Logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    scores = np.dot(X, theta)
    return np.sum(y*scores - np.log(1 + np.exp(scores)))

def logistic_regression(X, y, num_steps, learning_rate):
    weights = np.zeros(X.shape[1])

    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = y - predictions
        gradient = np.dot(X.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 1000 == 0:
            print("Loss: {}".format(log_likelihood(X, y, weights)))

    return weights

def predict(X, weights):
    return np.round(sigmoid(np.dot(X, weights)))

def accuracy(X_test, y_test, weights):
    n = X_test.shape[0]
    corrects = []
    for x, y in zip(X_test, y_test):
        y_pred = predict(x, weights)
        corrects.append(y_pred - y)

    corrects = np.array(corrects)
    total_preds = np.count_nonzero(corrects == 0)
    return total_preds/n


if __name__ == '__main__':
    X_test = c1_X
    y_test = [1]*c0_X.shape[0]

    weights = logistic_regression(X, y, 10000, 0.001)

    acc = accuracy(X_test, y_test, weights)
    print("The accuracy is: {}".format(acc))
