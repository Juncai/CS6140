import numpy as np
import Kernels


class svm():

    def __init__(self, kernel='poly', **kwargs):
        self.kernel = Kernels.Kernels(kernel)
        self.tol = 0.001 if 'tol' not in kwargs else kwargs['tol']
        self.c = 1. if 'C' not in kwargs else kwargs['C']
        self.support_vectors_ = None
        self.a = None
        self.features = None
        self.label = None
        self.w = None
        self.b = None
        self.e = None

    def fit(self, features, label):
        self.features = features
        self.label = label

        # TODO initialize a, w, b and precompute E?

        n, d = np.shape(features)
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            if examine_all:
                for i in range(n):
                    num_changed += self.examine_example(i)
            else:
                for i in self.non_bounded_indexes():
                    num_changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

    def examine_example(self, i):
        y_i = self.label[i]
        a_i = self.a[i]
        e_i = self.e[i]
        r_i = e_i * y_i
        if (r_i < - self.tol and a_i < self.c) or (r_i > self.tol and a_i > 0):
            if len(self.non_bounded_indexes()) > 1:
                j = np.argmax(e_i - self.e)
                if self.take_step(i, j):
                    return 1

            for j in self.non_bounded_indexes(random=True):  # TODO fix the randomly loop
                if self.take_step(i, j):
                    return 1

            for j in self.random_f_indexes():
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
        w = self.w
        s = y_i * y_j
        if y_i != y_j:
            l = max(0, a_j - a_i)
            h = min(self.c, a_j - a_i + self.c)
        else:
            l = max(0, a_i + a_j - self.c)
            h = min(self.c, a_i + a_j)
        if l == h:
            return False
        k_i_i = self.kernel.get_value(i, i)     # TODO need to implement
        k_j_j = self.kernel.get_value(j, j)     # TODO need to implement
        k_i_j = self.kernel.get_value(i, j)     # TODO need to implement
        n = k_i_i + k_j_j - 2 * k_i_j
        if n <= 0:
            return False

        # update a
        a_j_new = a_j + y_j * (e_i - e_j) / n

        a_j_new_clipped = a_j_new
        if a_j_new < l:
            a_j_new_clipped = l
        elif l <= a_j_new and a_j_new <= h:
            a_j_new_clipped = a_j_new
        else:
            a_j_new_clipped = h

        if abs(a_j - a_j_new_clipped) < e * (a_j + a_j_new_clipped + e):    # TODO what's e?
            return False

        a_i_new = a_i + s * (a_j - a_j_new_clipped)
        self.a[i] = a_i_new
        self.a[j] = a_j_new


        # update b
        b_i_new = b - e_i + (a_i - a_i_new) * y_i * k_i_i + (a_j - a_j_new) * y_j * k_i_j
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


        # TODO update w and E w.r.t the new a and b
        # w = sum(a_i * y_i * x_i)
        # e = w * x - y

        return True

    def predict(self, features):
        pass


    def random_f_indexes(self):
        return []

    def non_bounded_indexes(self, random=False):
        # TODO return a random shuffle of the origin index list
        return []