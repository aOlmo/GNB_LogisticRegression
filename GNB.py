import numpy as np
import pandas as pd

c_pos = 0
cs = [0, 1]
n_classes = 2
dataset_layout = ["attr1", "attr2", "attr3", "attr4", "class"]

for i, name in enumerate(dataset_layout):
    if name == "class":
        c_pos = i

def mean(vals):
    return np.sum(vals) / vals.shape[0]


def stddev(vals):
    n = vals.shape[0]
    m = mean(vals)
    v = (np.sum((vals-m)**2)/n)
    return np.sqrt(v)


def probability_calculation(X, m, std):
    exp = np.exp(-(((X-m)**2) / (2 * std**2)))
    mult = (1 / (np.sqrt(2 * np.pi) * std))
    return mult * exp


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

# GNB:
# - Calculate Prior
pri_c1, pri_c0 = n_c1/n_total_rows, n_c0/n_total_rows

# - Calculate Mean and Std for each attr in each class
m_v_dict = {}
for c in cs:
    m_v_dict[c] = {}
    c_X = c0_X if c == 0 else c1_X
    for i, attr in enumerate(dataset_layout[:c_pos]):
        m_v_dict[c][attr] = {
            "mean": mean(c_X[:, i]),
            "std": stddev(c_X[:, i])
        }

inp_vector = c0_X[7]
print(inp_vector)

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

# After the probabilities calculation, we will choose the one that is the largest

print(class_probs)

if class_probs[0] > class_probs[1]:
    print("Class 0")
else:
    print("Class 1")

# - Predict new samples
# TODO