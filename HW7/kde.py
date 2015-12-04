import Kernels
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel


class KDE():

    def __init__(self, kernel=c.EUCLIDIAN):
        self.kernel = Kernels.Kernels(kern_name=kernel)
        self.x = []
        self.y = []
        self.c_list = []
        self.c_x = []

    def fit(self, x, y):
        self.x = x
        self.y = y
        self._pre_compute_class()

    def predict(self, xx):
        pred = []
        for cur_xx in xx:
            r = np.zeros((len(self.c_list),))

            for ind, c_x_array in enumerate(self.c_x):
                # r[ind] = self.kernel.get_value(c_x_array, [cur_xx]).sum()
                tmp = rbf_kernel(c_x_array, [cur_xx])
                r[ind] = tmp.sum()


            pred.append(self.c_list[np.argmax(r)])
            # pred.append(self.c_list[np.argmin(r)])
        return pred

    def _pre_compute_class(self):
        c_ind_dict = {}
        for ind_yy, yy in enumerate(self.y):
            if yy not in c_ind_dict.keys():
                c_ind_dict[yy] = []
            c_ind_dict[yy].append(ind_yy)
        for k, v in c_ind_dict.items():
            self.c_list.append(k)
            cur_x = []
            for ind in v:
                cur_x.append(self.x[ind])
            self.c_x.append(np.array(cur_x))
