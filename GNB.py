import numpy as np
import pandas as pd

from io import StringIO
from sklearn.model_selection import KFold


# ================== Gathering the data ================== #
data_fname = 'data_banknote_authentication.txt'
df = pd.read_csv(data_fname, sep=',', header=None)
X = df.iloc[:, 0:4].values
y = df[4].values
# ================== ================== ================== #

three_f = KFold(n_splits=3, shuffle=True, random_state=0)
print(three_f)