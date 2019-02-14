from GNB import GNB
from LogReg import LogReg
from Dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt


def shuffle_training_sets_with_fraction(X, y, index):
    data = np.c_[X, y]
    np.random.shuffle(data)
    X_train = data[:, :-1]
    y_train = data[:, -1]

    frac_X_train = X_train[:index]
    frac_y_train = y_train[:index]

    return frac_X_train, frac_y_train

def exercise_5_2_2(d):
    X_train, X_test, y_train, y_test = d.get_train_test_sets()
    n = d.get_n_data()

    runs = 5
    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

    GNB_accs = [0] * len(fractions)
    LR_accs = [0] * len(fractions)

    for run in range(runs):
        print("[+]: Run {}".format(run))
        print("-----------")

        for i, fraction in enumerate(fractions):
            index = int(n * fraction)

            frac_X_train, frac_y_train = \
                shuffle_training_sets_with_fraction(X_train, y_train, index)

            GNB_classifier = GNB(frac_X_train, frac_y_train)
            LogReg_classifier = LogReg(frac_X_train, frac_y_train, 10000, 0.001)

            GNB_acc = GNB_classifier.accuracy(X_test, y_test)
            LR_acc = LogReg_classifier.accuracy(X_test, y_test)

            GNB_accs[i] += GNB_acc
            LR_accs[i] += LR_acc

            print("Fraction {}".format(fraction))
            print("GNB acc: {}".format(GNB_acc))
            print("LR acc: {} \n".format(LR_acc))

    GNB_accs = np.array(GNB_accs) / runs
    LR_accs = np.array(LR_accs) / runs

    plt.plot(fractions, GNB_accs, 'bo-', label='line 1', linewidth=2)
    plt.plot(fractions, LR_accs, 'ro-', label='line 2', linewidth=2)
    plt.show()

def exercise_5_2_3(d):
    gen_class = 1
    n_gen_samples = 400

    # Get the complete training and test sets with kfold = 3
    X_train, X_test, y_train, y_test = d.get_train_test_sets()
    GNB_generator = GNB(X_train, y_train)

    # Get the X and y samples
    X_gen_samples = GNB_generator.gen_n_samples(n_gen_samples, gen_class)
    y_gen_samples = [gen_class] * X_gen_samples.shape[0]

    # Create an auxiliary GNB classifier to get the mean and std of the new data
    aux_GNB = GNB(X_gen_samples, y_gen_samples, [gen_class])
    gen_m_std_dict = aux_GNB.get_m_std_dict()

    # Create another GNB to get the mean and std of the original data
    base_GNB = GNB(X_train, y_train)
    base_m_std_dict = base_GNB.get_m_std_dict()

    print("[+]: Mean and std for the dataset:")
    print("[+]: {}".format(base_m_std_dict[1]))
    print("[+]: Mean and std for the generated dataset:")
    print("[+]: {}".format(gen_m_std_dict[1]))

def exercise_4(d):
    n_steps = 10000
    lr = 0.001

    X_train, X_test, y_train, y_test = d.get_train_test_sets()

    X_train = np.array(
        [[1, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    LR = LogReg(X_train, y_train, n_steps, lr)
    acc = LR.accuracy(X_train, y_train)

    print("[+]: The weight vector is: {}".format(LR.get_weights()))
    print("The accuracy is: {}".format(acc))

if __name__ == '__main__':
    d = Dataset()

    print("============== Exercise 5.2.2 ============== ")
    exercise_5_2_2(d)
    print("============== Exercise 5.2.3 ============== ")
    exercise_5_2_3(d)
    print("==============   Exercise 4   ============== ")
    exercise_4(d)