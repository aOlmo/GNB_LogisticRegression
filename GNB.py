import numpy as np
from Dataset import Dataset

d = Dataset()
c_pos = d.get_c_pos()
pri_c0, pri_c1 = d.get_classes_prior()
cs = [0, 1]
dataset_layout = d.get_dataset_layout()

class GNB:
    def __init__(self, X_train, y_train, cs, m_std_dict=False):
        self.X_train = X_train
        self.y_train = y_train
        self.cs = cs
        self.m_std_dict = self.calculate_m_std_dict() if not m_std_dict else m_std_dict

    def get_m_std_dict(self):
        return self.m_std_dict

    def mean(self, vals):
        return np.sum(vals) / vals.shape[0]

    def stddev(self, vals):
        n = vals.shape[0]
        m = self.mean(vals)
        v = (np.sum((vals-m)**2)/(n-1))
        return np.sqrt(v)

    def probability_calculation(self, X, m, std):
        exp = np.exp(-(((X-m)**2) / (2 * std**2)))
        mult = (1 / (np.sqrt(2 * np.pi) * std))
        return mult * exp

    # - Calculate Mean and Std for each attr in each class
    def calculate_m_std_dict(self):
        m_std_dict = {}
        data_train = np.c_[self.X_train, self.y_train]
        for c in self.cs:
            m_std_dict[c]={}
            X_train_c = data_train[data_train[:, c_pos] == c][:, :c_pos]
            for i, attr in enumerate(dataset_layout[:c_pos]):
                m_std_dict[c][attr] = {
                    "mean": self.mean(X_train_c[:, i]),
                    "std": self.stddev(X_train_c[:, i])
                }

        return m_std_dict

    # Calculation of predictions
    def predict(self, inp_vector):
        class_probs = {}
        for c in self.m_std_dict:
            class_probs[c] = 1
            # Select the prior amongst the 2
            prior = pri_c0 if c == 0 else pri_c1
            # The input vector must have the same amount of values as attrs per class
            for i, attr in enumerate(self.m_std_dict[c]):
                attr_vals = self.m_std_dict[c][attr]
                mean, std = attr_vals['mean'], attr_vals['std']
                # Iterate through input vector
                x = inp_vector[i]
                # Assuming conditional independence
                class_probs[c] += np.log(self.probability_calculation(x, mean, std))

            class_probs[c] += np.log(prior)

        # We will choose the largest probability
        return 0 if class_probs[0] > class_probs[1] else 1

    # Calculation of the accuracy
    def accuracy(self, X_test, y_test):
        n = X_test.shape[0]
        corrects = []
        for x, y in zip(X_test, y_test):
            y_pred = self.predict(x)
            corrects.append(y_pred - y)

        corrects = np.array(corrects)
        total_preds = np.count_nonzero(corrects == 0)
        return total_preds/n

    # Function that generates new samples from the trained GNB and given class
    def gen_n_samples(self, n, cls):
        a = np.zeros([n])
        for attr in self.m_std_dict[cls]:
            curr_attr = self.m_std_dict[cls][attr]
            m, std = curr_attr['mean'], curr_attr['std']
            gen_val = np.random.normal(m, std, n)
            a = np.vstack((gen_val, a))

        # Remove the last one
        a = a[:a.shape[0]-1]
        return np.flip(a.T, 1)


if __name__ == '__main__':
    gen_class = 1

    X_train, X_test, y_train, y_test = d.get_train_test_sets()

    init_GNB = GNB(X_train, y_train, cs)

    X_gen_samples = init_GNB.gen_n_samples(400, gen_class)
    y_gen_samples = [gen_class] * X_gen_samples.shape[0]
    print(init_GNB.accuracy(X_gen_samples, y_gen_samples))

    new_GNB = GNB(X_gen_samples, y_gen_samples, [gen_class])
    gen_m_std_dict = new_GNB.get_m_std_dict()

    print(init_GNB.get_m_std_dict()[1])
    print(new_GNB.get_m_std_dict()[1])
