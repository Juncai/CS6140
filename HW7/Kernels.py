from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances

class Kernels():

    def __init__(self, kern_name):
        if kern_name == c.RBF:
            self.kernel_fun = rbf_kernel
        elif kern_name == c.LINEAR:
            self.kernel_fun = self.linear_kernel
        elif kern_name == c.EUCLIDIAN:
            # self.kernel_fun = self.euclidian_distance
            self.kernel_fun = euclidean_distances


    def get_value(self, i, j=None):
        if j is None:
            j = i
        return self.kernel_fun(i, j)

    def euclidian_distance(self, x, y=None):
        yy = x if y is None else y
        return np.sqrt(np.dot(x, np.transpose(x)) - 2 * np.dot(x, yy) + np.dot(yy, yy))

    def linear_kernel(self, x, y=None):
        '''

        :param x:
        :param y:
        :return:
        '''
        yy = x if y is None else y
        res = np.dot(x, np.transpose(yy))
        return res