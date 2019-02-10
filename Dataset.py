import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


class Dataset:
    def __init__(self):
        self.classes = [0, 1]
        self.dataset_layout = ["attr1", "attr2", "attr3", "attr4", "class"]
        self.class_position = self.get_c_pos()
        self.data_file_name = 'data_banknote_authentication.txt'
        self.data = self.get_data()
        self.k_fold = 3
        self.n = self.data.shape[0]

    # ================== Gathering the data ================== #
    def get_data(self):
        df = pd.read_csv(self.data_file_name, sep=',', header=None)
        return df.values

    def get_c_pos(self):
        for i, name in enumerate(self.dataset_layout):
            if name == "class":
                return i
        return None

    def get_dataset_layout(self):
        return self.dataset_layout

    def get_n_data(self):
        return self.n

    def get_train_test_sets(self):
        kfold = KFold(self.k_fold, True, 0)
        X = self.data[:, 0:self.class_position]
        y = self.data[:, self.class_position]

        for train_index, test_index in kfold.split(self.data):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def get_classes_prior(self):
        # Get count of 0s and 1s
        n_c1 = np.count_nonzero(self.data, axis=0)[self.class_position]
        n_c0 = abs(n_c1 - self.n)

        # - Calculate Prior
        pri_c1, pri_c0 = n_c1 / self.n, n_c0 / self.n

        return pri_c0, pri_c1


    # ================== ================== ================== #
