import Kernels
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel


class KDE():

    def __init__(self, kernel=c.EUCLIDEAN):
        self.kernel = Kernels.Kernels(kernel_name=kernel, is_sim=True)
        self.x = []
        self.y = []
        self.c_list = []
        self.c_x = []

    def fit(self, x, y):
        self.x = x
        self.y = y
        self._pre_compute_class_bak()

    def predict(self, xx):
        pred = []
        for cur_xx in xx:
            r = np.zeros((len(self.c_list),))

            for ind, c_x_array in enumerate(self.c_x):
                r[ind] = np.sum(self.kernel.get_value(c_x_array, [cur_xx])) / c_x_array.shape[0]
                # tmp = rbf_kernel(c_x_array, [cur_xx])
                # r[ind] = tmp.sum()


            pred.append(self.c_list[np.argmax(r)])
            # pred.append(self.c_list[np.argmin(r)])
        return pred

    # def _pre_compute_class(self):
    #     c_ind_list = []
    #     for i in range(10):
    #         self.c_list.append(i)
    #         c_ind_list.append([])
    #     for ind_yy, yy in enumerate(self.y):
    #         c_ind_list[yy].append(ind_yy)
    #     for ind_cil, ci in enumerate(c_ind_list):
    #         cur_x = []
    #         for ind_x in ci:
    #             cur_x.append(self.x[ind_x])
    #         self.c_x.append(np.array(cur_x))

    def _pre_compute_class_bak(self):
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
