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
n_c0 = c0_X.shape[0]
n_c1 = c1_X.shape[0]
# ================== ================== ================== #

# - Define the Logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def log_likelihood(X, y, theta):
    scores = np.dot(X, theta)
    ll = np.sum(y*scores - np.log(1 + np.exp(scores)))
    return ll

# lr = 0.001
# theta = np.zeros(X.shape[1])
# for i in range(10000):
#     z = np.dot(X, theta)
#     h = sigmoid(z)
#     theta -= lr * gradient_ascent(X, h, y)
#     if i % 1000:
#         print(loss(h, y))


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
            print(log_likelihood(X, y, weights))

    return weights


weights = logistic_regression(X, y, 10000, 0.001)

t = sigmoid(np.dot(c0_X[99], weights))
print(t)