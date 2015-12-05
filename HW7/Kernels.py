from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

class Kernels():

    def __init__(self, kern_name):
        if kern_name == c.RBF:
            self.kernel_fun = rbf_kernel
        elif kern_name == c.LINEAR:
            self.kernel_fun = self.linear_kernel
        elif kern_name == c.EUCLIDIAN:
            self.kernel_fun = self.euclidian_distance
        elif kern_name == c.COSINE:
            self.kernel_fun = self.cosine_distance
        elif kern_name == c.GAUSSIAN:
            self.kernel_fun = self.gaussian_similarity
        elif kern_name == c.POLY:
            self.kernel_fun = self.poly_similarity


    def get_value(self, x, y=None):
        if y is None:
            y = x

        n_row = x.shape[0]
        n_col = len(y)
        res = np.zeros((n_row, n_col))

        for ind_y, yy in enumerate(y):
            res[:, ind_y] = self.kernel_fun(x, yy)

        return res

    def euclidian_distance(self, x, y):
        return np.sqrt(np.square(x - y).sum(axis=1))


    def cosine_distance(self, x, y):
        num = np.dot(x, y)
        den = np.sqrt(np.square(x).sum(axis=1)) * np.sqrt(np.dot(y.transpose(), y))
        return 1 - num / den

    def gaussian_similarity(self, x, y):
        n = len(y)
        # c1 = (2 * np.pi) ** - (n / 2)
        c2 = - (1 / 2)
        # return 1- c1 * np.exp(c2 * np.square(x - y).sum(axis=1))
        return 1 - np.exp(c2 * np.square(x - y).sum(axis=1))

    def poly_similarity(self, x, y):
        cc = 0
        return np.square(np.dot(x, y) + cc)

    def linear_kernel(self, x, y=None):
        '''

        :param x:
        :param y:
        :return:
        '''
        yy = x if y is None else y
        res = np.dot(x, np.transpose(yy))
        return res