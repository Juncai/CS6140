from sklearn.metrics.pairwise import rbf_kernel


class Kernels():

    def __init__(self, kern_name):
        if kern_name == 'rbf':
            self.kernel_fun = rbf_kernel

    def get_value(self, i, j):
        return self.kernel_fun(i, j)