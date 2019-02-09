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
data_fname = 'data_banknote_authentication.txt'
df = pd.read_csv(data_fname, sep=',', header=None)
data = df.values

kfold = KFold(3, True, 0)
X = data[:, 0:c_pos]
y = data[:, c_pos]
n = data.shape[0]

for train_index, test_index in kfold.split(data):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# Get count of 0s and 1s
n_c1 = np.count_nonzero(data, axis=0)[c_pos]
n_c0 = abs(n_c1 - n)

# - Calculate Prior
pri_c1, pri_c0 = n_c1 / n, n_c0 / n
# ================== ================== ================== #


def mean(vals):
    return np.sum(vals) / vals.shape[0]


def stddev(vals):
    n = vals.shape[0]
    m = mean(vals)
    v = (np.sum((vals-m)**2)/(n-1))
    return np.sqrt(v)


def probability_calculation(X, m, std):
    exp = np.exp(-(((X-m)**2) / (2 * std**2)))
    mult = (1 / (np.sqrt(2 * np.pi) * std))
    return mult * exp


# GNB:
# - Calculate Mean and Std for each attr in each class
def calculate_m_v_dict(X_train, y_train):
    m_v_dict = {}
    data_train = np.c_[X_train, y_train]
    for c in cs:
        m_v_dict[c]={}
        X_train_c = data_train[data_train[:, c_pos] == c][:, :c_pos]
        for i, attr in enumerate(dataset_layout[:c_pos]):
            m_v_dict[c][attr] = {
                "mean": mean(X_train_c),
                "std": stddev(X_train_c)
            }

    return m_v_dict

# Calculation of predictions
def predict(inp_vector, m_v_dict):
    class_probs = {}
    for c in m_v_dict:
        class_probs[c] = 1
        # Select the prior amongst the 2
        prior = pri_c0 if c == 0 else pri_c1
        # The input vector must have the same amount of values as attrs per class
        for i, attr in enumerate(m_v_dict[c]):
            attr_vals = m_v_dict[c][attr]
            mean, std = attr_vals['mean'], attr_vals['std']
            # Iterate through input vector
            x = inp_vector[i]
            # Assuming conditional independence
            class_probs[c] += np.log(probability_calculation(x, mean, std))

        class_probs[c] += np.log(prior)

    # We will choose the largest probability
    return 0 if class_probs[0] > class_probs[1] else 1


# Calculation of the accuracy
def accuracy(X_test, y_test, m_v_dict):
    n = X_test.shape[0]
    corrects = []
    for x, y in zip(X_test, y_test):
        y_pred = predict(x, m_v_dict)
        corrects.append(y_pred - y)

    corrects = np.array(corrects)
    total_preds = np.count_nonzero(corrects == 0)
    return total_preds/n

# Function that generates new samples from the trained GNB and given class
def gen_n_samples(n, c, m_v_dict):
    ret = []
    a = np.zeros([n])
    for attr in m_v_dict[c]:
        curr_attr = m_v_dict[c][attr]
        m, std = curr_attr['mean'], curr_attr['std']
        gen_val = np.random.normal(m, std, n)
        a = np.vstack((gen_val, a))

    # Remove the last one
    a = a[:a.shape[0]-1]
    return np.flip(a.T, 1)


if __name__ == '__main__':

    m_v_dict = calculate_m_v_dict(X_train, y_train)
    # acc = accuracy(X_test, y_test, m_v_dict)
    # print("The accuracy is: {}".format(acc))

    X_test = gen_n_samples(400, 0, m_v_dict)
    y_test = [0] * X_test.shape[0]
    print(accuracy(X_test, y_test, m_v_dict))
