import Kernels
import numpy as np
import Consts as c
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel


class kNN():

    def __init__(self, kernel=c.EUCLIDIAN):


        self.kernel = Kernels.Kernels(kern_name=kernel)
        self.x = []
        self.y = []

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, xx, k=None, r=None):
        if k is not None and r is not None:
            raise Exception('Ambiguous parameters with both k and range specified.')
        if k is None and r is None:
            raise Exception('Either k or range should be specified')
        if k is not None:  # kNN
            dists = self.kernel.get_value(self.x, xx)
            # dists = rbf_kernel(self.x, xx)
            n_ind = self.find_n_nearest(k, dists)
            n = xx.shape[0]
            pred = np.zeros((n,))
            for i in range(n):
                pred[i] = self.vote(n_ind[i])
            return pred
        else:
            dists = self.kernel.get_value(self.x, xx)
            n_ind = self.find_n_within_radius(r, dists)
            n = xx.shape[0]
            pred = np.zeros((n,))
            for i in range(n):
                pred[i] = self.vote(n_ind[i])
            return pred


    def vote(self, inds):
        vote_dict = {}
        for ind in inds:
            if self.y[ind] not in vote_dict:
                vote_dict[self.y[ind]] = 1
            else:
                vote_dict[self.y[ind]] += 1

        max = 0
        final_label = None
        for k, v in vote_dict.items():
            if v > max:
                max = v
                final_label = k
        return final_label

    def find_n_nearest(self, n, dists):
        res = []
        nn = dists.shape[1]
        for i in range(nn):
            dist = dists[:, i]
            res.append(self.find_n_nearest_helper(n, dist))
        return res

    def find_n_nearest_helper(self, n, dist):
        assert isinstance(n, int)
        n = n if n < len(dist) else len(dist)
        if n < 1:
            raise Exception('n should be an integer larger than 0')
        inds = None
        if n == 1:
            inds = np.array([np.argmin(dist)])
        else:
            inds = dist.argsort()[:n]
        return inds

    def find_n_within_radius(self, r, dists):
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
