import Kernels
import numpy as np
import Consts as c
import Utilities as util
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel


class kNN():

    def __init__(self, kernel=c.EUCLIDEAN):
        self.kernel = Kernels.Kernels(kern_name=kernel)
        self.x = []
        self.y = []
        self.w = []
        self.f_select = False
        self.best_f_indices = []

    def fit(self, x, y, f_select=False, best_f=5):
        self.x = x
        self.y = y
        if f_select:
            self.f_select = True
            self._calculate_w()
            self.best_f_indices = util.find_top_indices(best_f, self.w, False)
            self.x = self.x[:, self.best_f_indices]

    def _calculate_w(self):
        n, f_n = self.x.shape
        self.w = np.zeros((f_n,))
        dists = self.kernel.get_value(self.x)
        for x_ind, xx in enumerate(self.x):
            # find nearest same class data point
            # find nearest different class data point
            z_same = None
            z_opp = None
            min_dist_same = float('inf')
            min_dist_opp = float('inf')
            for d_ind, d in enumerate(dists[x_ind]):
                # if x_ind == d_ind:
                #     continue
                if self.y[d_ind] == self.y[x_ind]:
                    if d < min_dist_same:
                        z_same = self.x[d_ind]
                        min_dist_same = d
                else:
                    if d < min_dist_opp:
                        z_opp = self.x[d_ind]
                        min_dist_opp = d
            self.w = self.w - np.square(xx - z_same) + np.square(xx - z_opp)

    def predict(self, xx, k=None, r=None):
        if k is not None and r is not None:
            raise Exception('Ambiguous parameters with both k and range specified.')
        if k is None and r is None:
            raise Exception('Either k or range should be specified')

        if self.f_select:
            xx = xx[:, self.best_f_indices]

        dists = self.kernel.get_value(self.x, xx)
        if k is not None:  # kNN
            # dists = rbf_kernel(self.x, xx)
            n_ind = self.find_n_nearest(k, dists)
            n = xx.shape[0]
            pred = np.zeros((n,))
            for i in range(n):
                # pred[i] = self.vote(n_ind[i]) if self.ds == c.DS_DIGITS else self.vote_bak(n_ind[i], self.y)
                pred[i] = self.vote(n_ind[i])
            return pred
        else:
            n_ind = self.find_n_within_radius(r, dists)
            n = xx.shape[0]
            pred = np.zeros((n,))
            for i in range(n):
                pred[i] = self.vote(n_ind[i])
            return pred


    def vote(self, inds):
        accu = np.zeros((10,))
        for ind in inds:
            accu[self.y[ind]] += 1
        return np.argmax(accu)




    def vote_bak(self, inds, y):
        vote_dict = {}
        for ind in inds:
            if y[ind] not in vote_dict:
                vote_dict[y[ind]] = 1
            else:
                vote_dict[y[ind]] += 1

        max = 0
        final_label = None
        for k, v in vote_dict.items():
            if v > max:
                max = v
                final_label = k
        return final_label

    def find_n_nearest(self, n, dists):
        res = []
        for d in dists:
            res.append(util.find_top_indices(n, d))
        return res

    def find_n_nearest_bak(self, n, dists):
        res = []
        nn = dists.shape[1]
        for i in range(nn):
            dist = dists[:, i]
            res.append(util.find_top_indices(n, dist))
        return res



    def find_n_within_radius(self, r, dists):
        res = []
        for d in dists:
            res.append(self.find_n_within_radius_helper(r, d))
        return res

    def find_n_within_radius_bak(self, r, dists):
        res = []
        nn = dists.shape[1]
        for i in range(nn):
            dist = dists[:, i]
            res.append(self.find_n_within_radius_helper(r, dist))
        return res

    def find_n_within_radius_helper(self, r, dist):
        res = []
        for d_ind, d in enumerate(dist):
            if d <= r:
                res.append(d_ind)
        return res

if __name__ == '__main__':
    knn = kNN()
    # dist = [1, 2, 3, 4, 5, 6, 0]
    # r = 3
    # print(knn.find_n_within_radius_helper(r, dist))

    inds = [2, 6, 7, 0]
    y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    res = knn.vote_bak(inds, y)
    print(res)

