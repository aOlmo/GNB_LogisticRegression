import numpy as np
import pandas as pd

from io import StringIO
from sklearn.model_selection import KFold


def mean(vals):
    return np.sum(vals) / vals.shape[0]

def stddev(vals):
    n = vals.shape[0]
    m = mean(vals)
    v = (np.sum((vals-m)**2)/n)
    return np.sqrt(v)

def PDF(X, m, std):
    exp = np.exp(-(((X-m)**2) / (2 * std**2)))
    mult = (1 / (np.sqrt(2 * np.pi) * std))
    return mult * exp

dataset_layout = ["attr1", "attr2", "attr3", "attr4", "class"]

for i, name in enumerate(dataset_layout):
    if name == "class":
        c_pos = i

n_classes = 2
cs = [0, 1]

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
pri_c_1, pri_c0 = n_c1/n_total_rows, n_c0/n_total_rows

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

print(m_v_dict[0])

# - Create PDF

# - Predict new samples

# # ====================== Three-fold ====================== #
# three_f = KFold(n_splits=3, shuffle=False, random_state=0)
# for train_index, test_index in three_f.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# # ================== ================== ================== #
#
