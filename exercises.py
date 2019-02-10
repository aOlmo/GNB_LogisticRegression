from Dataset import Dataset
from GNB import GNB
from LogReg import LogReg

import matplotlib.pyplot as plt

if __name__ == '__main__':
    d = Dataset()
    n = d.get_n_data()
    X_train, X_test, y_train, y_test = d.get_train_test_sets()

    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

    for fraction in fractions:
        index = int(n*fraction)

        X_train = X_train[:index]
        y_train = y_train[:index]

        print(X_train)
        print()
        print(y_train)
        exit()

        GNB_classifier = GNB(X_train, y_train)
        LogReg_classifier = LogReg(X_train, y_train, 10000, 0.001)

        GNB_acc = GNB_classifier.accuracy(X_test, y_test)
        LR_acc = LogReg_classifier.accuracy(X_test, y_test)

        print("Fraction {}".format(fraction))
        print("GNB acc: {}".format(GNB_acc))
        print("LR acc: {}".format(LR_acc))
        print()


