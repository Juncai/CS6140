from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


class Kernels():

    def __init__(self, kern_name, is_sim=True):
        self.is_sim = is_sim
        if kern_name == c.LINEAR:
            self.kernel_fun = self.linear
        elif kern_name == c.EUCLIDEAN:
            self.kernel_fun = self.euclidean
        elif kern_name == c.COSINE:
            self.kernel_fun = self.cosine
        elif kern_name == c.GAUSSIAN:
            self.kernel_fun = self.gaussian
        elif kern_name == c.POLY:
            self.kernel_fun = self.poly
        elif kern_name == c.RBF:
            self.kernel_fun = self.rbf

    def get_value_bak(self, x, y=None):
        if y is None:
            y = x

        res = []
        for yy in y:
            res.append(self.kernel_fun(x, yy))

        return res

    def get_value(self, x, y=None):
        if self.kernel_fun == self.rbf:
            if y is None:
                y = x
            return rbf_kernel(x, y)
        if y is None:
            y = x

        n_row = x.shape[0]
        n_col = len(y)
        res = np.zeros((n_row, n_col))

        for ind_y, yy in enumerate(y):
            res[:, ind_y] = self.kernel_fun(x, yy)

        return res

    def euclidean(self, x, y):
        """
        (sum((x - y)^2))^0.5
        :return:
        """
        res = np.sqrt(np.square(x - y).sum(axis=1))
        return np.exp(-res) if self.is_sim else res


    def cosine(self, x, y):
        """
        (x * y) / (((x^2)^0.5) * ((y^2)^0.5))
        :return:
        """
        num = np.dot(x, y)
        den = np.sqrt(np.square(x).sum(axis=1)) * np.sqrt(np.dot(y.transpose(), y))
        res = num / den
        return res if self.is_sim else 1 - res

    def gaussian(self, x, y):
        """

        :return:
        """
        # c2 = - (1 / 2)
        c2 = -1
        res = np.exp(c2 * np.square(x - y).sum(axis=1))
        return res if self.is_sim else - res



    def poly(self, x, y):
        cc = 0
        res = np.square(np.dot(x, y) + cc)
        return res if self.is_sim else 1 / res


    def linear(self, x, y=None):
        '''

        :param x:
        :param y:
        :return:
        '''
        yy = x if y is None else y
        res = np.dot(x, np.transpose(yy))
        # return res if self.is_sim else np.exp(-res)
        return res

    def rbf(self, x, y):
        """

        :return:
        """
        n = len(y)
        # c2 = - (1 / 2)
        c2 = -1
        # res = np.exp(c2 * np.square(x - y).sum(axis=1))

        res = rbf_kernel(x, y)[:, 0]
        return res if self.is_sim else - res