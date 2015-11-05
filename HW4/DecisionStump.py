import random
import Model
import Utilities as util
import copy
import numpy as np

class DecisionStump(Model.Model):

    def __init__(self, ecoc=None):
        self.f_ind = None
        self.thresh = None
        self.w_err = None
        # self.n_err = None   # non weighted err
        self.ecoc = ecoc

    def predict_single(self, feature):
        if self.ecoc is not None:
            f_val = feature[self.f_ind] if self.f_ind in feature.keys() else 0
            return 1 if f_val >= self.thresh else -1
        else:
            if not isinstance(self.thresh, tuple):
                return 1 if feature[self.f_ind] >= self.thresh else -1
            else:
                return 1 if feature[self.f_ind] == self.thresh[1] else -1

    def train(self, features, label, d, threshes, thresh_cs=None):
        self.f_ind, self.thresh, self.w_err = self._split_on_err(features, label, d, threshes, thresh_cs)

    def _split_on_err(self, features, label, d, threshes, thresh_cs):
        if self.ecoc is not None:
            return self._split_on_err_ecoc(features, label, d, threshes)
        else:
            return self._split_on_err_normal(features, label, d, threshes, thresh_cs)

    def _split_on_err_ecoc(self, features, label, d, threshes):

        best_res = None
        max = 0     # max value of 1/2-error(h)
        n = len(features)
        n_ones = np.ones((n, 1))
        label_plus_one = np.array(label) + 1
        # calculate weighted errors for other thresholds
        for t_k in threshes.keys():
            cur_f = np.array([(f[t_k] if t_k in f.keys() else 0) for f in features])
            for t in threshes[t_k]:
                cur_r = np.sign(cur_f - t) + 1
                cur_r = np.logical_xor(cur_r, label_plus_one)
                w_err = np.dot(cur_r, d)
                n_err = np.dot(cur_r, n_ones)
                err_d = abs(0.5 - w_err)
                if err_d > max:
                    max = err_d
                    best_res = (t_k, t, w_err, n_err/n)
        return best_res


    def _split_on_err_normal(self, features, label, d, threshes, thresh_cs):
        # TODO need to handle the discrete features
        '''
        Find the best pair based on IG
        Return: feature index, threshold, left predict, right predict
        '''
        best_res = None
        max = 0 # max value of 1/2-error(h)
        # n = len(features)
        # n_ones = np.ones((1, n)).tolist()[0]
        # TODO deal with the discrete thresholds
        if isinstance(threshes[0][0], bool):
            # threshes_cs = util.pre_compute_threshes_discrete(features, label, threshes, d)
            for i in range(len(features[0])):
                for j, t_cs in enumerate(thresh_cs[i]):
                    w_err = np.dot(t_cs, d)
                    err_d = abs(0.5 - w_err)
                    if err_d > max:
                        max = err_d
                        best_res = (i, (threshes[i][0], threshes[i][1][j]), w_err)
        else:
            # threshes_cs = util.pre_compute_threshes_3(features, label, threshes, d)
            for i, _ in enumerate(features[0]):
                for j, t_cs in enumerate(thresh_cs[i]):
                    w_err = np.dot(t_cs, d)
                    # n_err = np.dot(t_cs, n_ones) / n
                    # w_err, n_err = threshes_cs[i][j]
                    err_d = abs(0.5 - w_err)
                    if err_d > max:
                        max = err_d
                        best_res = (i, threshes[i][j], w_err)
        return best_res

    def weighted_error(self, ind, features, label, d, thresh):
        '''
        Compute weighted error
        '''
        n = len(features)
        w_e = 0.
        for i in range(n):
            if features[i][ind] < thresh:
                if label[i] == 1:
                    w_e += d[i]
            else:
                if label[i] == -1:
                    w_e += d[i]
        return w_e


class RandomDecisionStump(DecisionStump):

    def _split_on_err(self, features, label, d, threshes, thresh_cs):
        f_ind = random.randint(0, len(threshes) - 1)
        thresh_ind = random.randint(0, len(threshes[f_ind]) - 1)
        thresh = threshes[f_ind][thresh_ind]
        err = self.weighted_error(f_ind, features, label, d, thresh)
        return f_ind, thresh, err