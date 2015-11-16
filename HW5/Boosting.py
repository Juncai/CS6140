import Model
import numpy as np
import math


class Boosting():

    def __init__(self, distribution):
        self.model = []
        self.d = distribution
        self.a = []


    def test(self, features, labels, err_fun):
        return err_fun(self.predict(features), labels, 0)

    def predict(self, features, is_raw=False):
        preds = []
        for f in features:
            preds.append(self.predict_single(f, is_raw))
        return preds


    def predict_single(self, feature, is_raw=False):
        res = 0.
        for i in range(len(self.model)):
            pred = self.model[i].predict_single(feature)
            tmp = pred * self.a[i]
            res += tmp
        return math.copysign(1, res) if not is_raw else res

    def update_predict(self, features, pre_pred):
        '''

        :param features:
        :param pre_pred:
        '''
        n = len(features)
        for i, f in enumerate(features):
            pred = self.predict_with_model_single(-1, f)
            pre_pred[i] += pred


    def predict_with_model_single(self, ind, feature):
        '''

        :param ind: model index
        :param feature:
        :return: predict value of specified model
        '''
        pred = self.model[ind].predict_single(feature)
        return pred * self.a[ind]

    def add_model(self, m_class, features, label, threshes, thresh_cs, is_uniform=False):
        m = m_class(is_uniform)
        m.train(features, label, self.d, threshes, thresh_cs)
        alpha = self.compute_alpha(m.w_err)
        self.a.append(alpha)
        if thresh_cs is not None:
            self.update_d_cs(thresh_cs, m, alpha)
        else:
            self.update_d(features, label, m, alpha)
        self.model.append(m)


    def compute_alpha(self, err):
        if err < 0.005:
            err = 0.005
        if err > 0.995:
            err = 0.995
        tmp = (1 - err) / (err)
        alpha = 0.5 * math.log(tmp, math.e)
        return alpha

    def update_d(self, features, label, model, a):
        for i in range(len(self.d)):
            y = label[i]
            pred = model.predict_single(features[i])
            self.d[i] *= math.exp(-a * y * pred)

        # TODO calculate Zt
        zt = math.fsum(self.d)
        # update distribution
        self.d = self.d / zt

    def update_d_cs(self, thresh_cs, model, a):
        for i in range(len(self.d)):
            self.d[i] *= math.exp(a) if thresh_cs[model.f_ind][model.t_ind][i] else math.exp(-a)

        # TODO calculate Zt
        zt = math.fsum(self.d)
        # update distribution
        self.d = self.d / zt
