import numpy as np
import random
# from sklearn.metrics.pairwise import rbf_kernel
import Kernels
import time


class SVM():

    def __init__(self, **kwargs):
        self.tol = 0.01 if 'tol' not in kwargs else kwargs['tol']
        self.c = 1. if 'C' not in kwargs else kwargs['C']
        self.epsilon = 0.001 if 'epsilon' not in kwargs else kwargs['epsilon']
        self.kernel_fun = Kernels.Kernels('linear') if 'kernel' not in kwargs else Kernels.Kernels(kwargs['kernel'])
        self.support_vectors_ = []
        self.a = []
        self.ay = []
        self.features = []
        self.label = []
        self.b = []
        self.e = []
        self.acc = 0
        self.kernel = []

    def fit(self, features, label):
        self.features = features
        self.label = label

        n, d = np.shape(features)

        # TODO initialize a, w, b, kernel and precompute E?
        # self.a = np.random.rand(n) * self.c   # randomly initialize the lagrangian multipliers
        self.a = self.init_a()
        # init_a = 0.1    # TODO find a proper initial value for lm
        # self.a = (np.ones((1, n)) * init_a)[0]  # initialize the lm

        # self.b = random.random()
        self.b = 0
        self.kernel = self.kernel_fun.get_value(features)

        self.ay = self.a * self.label

        # TODO calculate predictions
        fx = np.dot(self.ay, self.kernel) + self.b
        self.e = fx - self.label

        # start training
        # while not self.converged:
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(n):
                    num_changed += self.examine_example(i)
            else:
                for i in self.non_bounded_indices():
                    num_changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            f_x = np.dot(self.ay, self.kernel) + self.b
            pred = np.sign(f_x)
            acc = (pred == self.label).sum() / len(self.label)
            print('{} Training acc: {}, num_changed: {}'.format(time.time(), acc, num_changed))

    def examine_example(self, i):
        y_i = self.label[i]
        a_i = self.a[i]
        e_i = self.e[i]
        r_i = e_i * y_i
        if (r_i < - self.tol and a_i < self.c) or (r_i > self.tol and a_i > 0):
            nb_indices = self.non_bounded_indices()
            if len(nb_indices) > 1:
                j = np.argmax(np.abs(e_i - self.e))
                if self.take_step(i, j):
                    return 1

            np.random.shuffle(nb_indices)
            for j in nb_indices:  # TODO fix the randomly loop
                if self.take_step(i, j):
                    return 1

            rf_indices = self.random_f_indices()
            for j in rf_indices:
                if self.take_step(i, j):
                    return 1
        return 0


    def take_step(self, i, j):
        if i == j:
            return False
        y_i = self.label[i]
        a_i = self.a[i]
        e_i = self.e[i]
        y_j = self.label[j]
        a_j = self.a[j]
        e_j = self.e[j]
        b = self.b
        c = self.c
        s = y_i * y_j
        if y_i != y_j:
            l = max(0, a_j - a_i)
            h = min(c, a_j - a_i + c)
        else:
            l = max(0, a_i + a_j - c)
            h = min(c, a_i + a_j)
        if l == h:
            return False
        k_i_i = self.kernel[i][i]
        k_j_j = self.kernel[j][j]
        k_i_j = self.kernel[i][j]
        k_j_i = self.kernel[j][i]
        n = k_i_i + k_j_j - 2 * k_i_j
        if n <= 0:
            return False

        # update a
        a_j_new = a_j + y_j * (e_i - e_j) / n

        if a_j_new < l:
            a_j_new_clipped = l
        elif l <= a_j_new and a_j_new <= h:
            a_j_new_clipped = a_j_new
        else:
            a_j_new_clipped = h

        a_j_new = a_j_new_clipped

        if abs(a_j - a_j_new) < self.epsilon * (a_j + a_j_new + self.epsilon):
            return False

        a_i_new = a_i + s * (a_j - a_j_new)
        self.a[i] = a_i_new
        self.a[j] = a_j_new
        self.ay[i] = a_i_new * self.label[i]
        self.ay[j] = a_j_new * self.label[j]

        # update b TODO confirm the e_i and e_j is old value or not
        # # update E w.r.t the new a
        # f_x = np.dot(self.ay, self.kernel) + b
        # self.e = f_x - self.label
        # e_i = self.e[i]
        # e_j = self.e[j]

        b_i_new = b - e_i + (a_i - a_i_new) * y_i * k_i_i + (a_j - a_j_new) * y_j * k_j_i
        b_j_new = b - e_j + (a_i - a_i_new) * y_i * k_i_j + (a_j - a_j_new) * y_j * k_j_j
        if 0 < a_i_new and a_i_new < self.c:
                # if both are non-bound, choose one of the b_new, here we always choose the b_i_new
                # only i is non-bound, choose b_i_new
                b_new = b_i_new
        elif 0 < a_j_new and a_j_new < self.c:
            b_new = b_j_new
        else:
            b_new = (b_i_new + b_j_new) / 2
        self.b = b_new

        # update E w.r.t the new a and b
        f_x = np.dot(self.ay, self.kernel) + self.b
        self.e = f_x - self.label

        # self.e = self.e - b + self.b
        # self.e[i] = np.dot(self.ay, self.kernel[:, i]) + self.b
        # self.e[j] = np.dot(self.ay, self.kernel[:, j]) + self.b

        # acc = (np.sign(f_x * self.label) + 1).mean() / 2
        # pred = np.sign(f_x)
        # acc = (pred == self.label).sum() / len(self.label)
        # print('{} Training acc: {}'.format(time.time(), acc))

        return True

    def predict(self, features):
        pred = np.dot(self.ay, self.kernel_fun.get_value(self.features, features)) + self.b
        return np.sign(pred)


    def random_f_indices(self):
        f_ind = np.array([i for i in range(len(self.features))])
        np.random.shuffle(f_ind)
        return f_ind

    def non_bounded_indices(self, random=False):
        # TODO return a random shuffle of the origin index list
        # non_bounded = 0 < a_i < C
        nb_ind = np.array([i for i, aa in enumerate(self.a) if aa > 0 and aa < self.c])
        if random:
            np.random.shuffle(nb_ind)
        return nb_ind

    def init_a(self):
        n = len(self.label)
        res = np.zeros((1, n))[0]

        # pos_indices = [i for i, y in enumerate(self.label) if y == 1]
        # neg_indices = [i for i, y in enumerate(self.label) if y == -1]
        #
        # np.random.shuffle(pos_indices)
        # np.random.shuffle(neg_indices)
        #
        # nn = len(pos_indices) if len(pos_indices) < len(neg_indices) else len(neg_indices)
        #
        # for i in range(nn):
        #     res[pos_indices[i]] = 0.2
        #     res[neg_indices[i]] = 0.2

        return res