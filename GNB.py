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

# - Calculate Prior
pri_c1, pri_c0 = n_c1 / n_total_rows, n_c0 / n_total_rows
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
def calculate_m_v_dict():

    m_v_dict = {}
    for c in cs:
        m_v_dict[c] = {}
        c_X = c0_X if c == 0 else c1_X
        for i, attr in enumerate(dataset_layout[:c_pos]):
            m_v_dict[c][attr] = {
                "mean": mean(c_X[:, i]),
                "std": stddev(c_X[:, i])
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

# Function that generates new samples from the trained GNB
def gen_n_samples(n, c, m_v_dict):
    ret = []
    a = np.zeros([n])
    for attr in m_v_dict[c]:
        curr_attr = m_v_dict[c][attr]
        m, std = curr_attr['mean'], curr_attr['std']
        gen_val = np.random.normal(m, std, n)
        a = np.vstack((gen_val, a))
        print(attr+str(gen_val))

    # Remove the last one
    a = a[:a.shape[0]-1]
    return np.flip(a.T, 1)



if __name__ == '__main__':
    X_test = c1_X
    y_test = [1] * X_test.shape[0]  # TODO!

    m_v_dict = calculate_m_v_dict()
    # acc = accuracy(X_test, y_test, m_v_dict)
    # print("The accuracy is: {}".format(acc))
    # TODO: improve this
    test = gen_n_samples(400, 1, m_v_dict)
    for i in test:
        print(predict(i, m_v_dict))