import numpy as np



class svm():

    def __init__(self, kernel='poly', **kwargs):
        self.kernel = kernel
        self.tol = 0.001 if 'tol' not in kwargs else kwargs['tol']
        self.c = 1. if 'C' not in kwargs else kwargs['C']
        self.support_vectors_ = None
        self.a = None

    def fit(self, features, label):
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

        pass

    def predict(self, features):
        pass

    def non_bounded_indexes(self):
        return []