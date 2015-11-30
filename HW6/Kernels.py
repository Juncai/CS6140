from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


class Kernels():

    def __init__(self, kern_name):
        if kern_name == 'rbf':
            self.kernel_fun = rbf_kernel
        elif kern_name == 'linear':
            self.kernel_fun = self.kernel_fun


    def get_value(self, i, j=None):
        if j is None:
            j = i
        return self.kernel_fun(i, j)


    def linear_kernel(self, x, y=None):
        '''

        :param x:
        :param y:
        :return:
        '''
        yy = x if y is None else y
        res = np.dot(x, np.transpose(yy))
        return res