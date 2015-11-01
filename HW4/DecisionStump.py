import random
import Model
import Utilities as util

class DecisionStump(Model.Model):
    f_ind = None
    thresh = None
    w_err = None
    n_err = None    # non weighted err

    def __init__(self):
        self.f_ind = None
        self.thresh = None

    def predict_single(self, feature):
        if not isinstance(self.thresh, tuple):
            return 1 if feature[self.f_ind] > self.thresh else -1
        else:
            return 1 if feature[self.f_ind] == self.thresh[1] else -1

    def train(self, features, label, d, threshes):
        self.f_ind, self.thresh, self.w_err, self.n_err = self._split_on_err(features, label, d, threshes)

    def _split_on_err(self, features, label, d, threshes):
        # TODO need to handle the discrete features
        '''
        Find the best pair based on IG
        Return: feature index, threshold, left predict, right predict
        '''
        best_res = None
        max = 0 # max value of 1/2-error(h)
        # TODO deal with the discrete thresholds
        if isinstance(threshes[0][0], bool):
            threshes_cs = util.pre_compute_threshes_discrete(features, label, threshes, d)
            for i in range(len(features[0])):
                for j in range(len(threshes_cs[i])):
                    w_err, n_err = threshes_cs[i][j]
                    err_d = abs(0.5 - w_err)
                    if err_d >= max:
                        max = err_d
                        best_res = (i, (threshes[0][0], threshes[i][1][j]), w_err, n_err)
        else:
            threshes_cs = util.pre_compute_threshes_3(features, label, threshes, d)
            for i in range(len(features[0])):
                for j in range(len(threshes_cs[i])):
                    w_err, n_err = threshes_cs[i][j]
                    err_d = abs(0.5 - w_err)
                    if err_d >= max:
                        max = err_d
                        best_res = (i, threshes[i][j], w_err, n_err)
        return best_res

    def weighted_error(self, ind, features, label, d, thresh):
        '''
        Compute weighted error
        '''
        n = len(features)
        w_e = 0.
        n_e = 0.
        for i in range(n):
            if features[i][ind] <= thresh:
                if label[i] == 1:
                    w_e += d[i]
                    n_e += 1
            else:
                if label[i] == -1:
                    w_e += d[i]
                    n_e += 1
        return w_e, n_e / n


class RandomDecisionStump(DecisionStump):

    def _split_on_err(self, features, label, d, threshes):
        f_ind = random.randint(0, len(threshes) - 1)
        thresh_ind = random.randint(0, len(threshes[f_ind]) - 1)
        thresh = threshes[f_ind][thresh_ind]
        err = self.weighted_error(f_ind, features, label, d, thresh)
        return f_ind, thresh, err[0], err[1]