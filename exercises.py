from Dataset import Dataset
from GNB import GNB
from LogReg import LogReg

import matplotlib.pyplot as plt

if __name__ == '__main__':
    d = Dataset()
    n = d.get_n_data()
    X_train, X_test, y_train, y_test = d.get_train_test_sets()

    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

    GNB_accs = []
    LR_accs = []
    for fraction in fractions:
        index = int(n*fraction)

        frac_X_train = X_train[:index]
        frac_y_train = y_train[:index]

        GNB_classifier = GNB(frac_X_train, frac_y_train)
        LogReg_classifier = LogReg(frac_X_train, frac_y_train, 10000, 0.001)

        GNB_acc = GNB_classifier.accuracy(X_test, y_test)
        LR_acc = LogReg_classifier.accuracy(X_test, y_test)

        GNB_accs.append(GNB_acc)
        LR_accs.append(LR_acc)

        print("Fraction {}".format(fraction))
        print("GNB acc: {}".format(GNB_acc))
        print("LR acc: {} \n".format(LR_acc))

    plt.plot(fractions, GNB_accs, 'bo-', label='line 1', linewidth=2)
    plt.plot(fractions, LR_accs, 'ro-', label='line 2', linewidth=2)
    plt.show()



