import numpy as np
import numpy.random as random
import Boosting as b


class DataSet():

    def __init__(self, datapoints):
        self.data = datapoints
        # index list of member elements
        self.n = len(datapoints[0])
        self.all = [i for i in range(self.n)]
        self.tr = []
        self.te = []
        self.re = []

    def random_pick(self, c):
        '''
        Randomly pick data points from the dataset
        :param c:
        :return:
        '''
        self._init_elements()
        size = int(self.n * c / 100)
        self.tr = random.choice(self.all, size, replace=False)
        self.re = list(set(self.all) - set(self.tr))
        return self.get_data(self.tr)

    def active_pick(self, c, model):
        '''

        :param c:
        :param model:
        :return:
        '''
        num = int(self.n * c / 100)
        assert isinstance(model, b.Boosting)
        re = self.get_data(self.re)
        re_preds = model.predict(re[0])
        new_tr = self.get_low_conf_dp(num, re_preds, 0)
        # update the training and remaining data
        self.r = list(set(self.r) - set(new_tr))
        self.tr = list(set(self.tr).union(set(new_tr)))
        return self.get_training()

    def get_low_conf_dp(self, num, pred, ss=0):
        res_ind = []

        res = np.abs(np.array(pred) - ss).tolist()
        res = [[r, i] for i, r in enumerate(res)]
        res = sorted(res, key=lambda r:r[0], reverse=False)
        for i in range(num):
            res_ind.append(res[i][1])
        return res_ind


    def get_data(self, member_index):
        res = [[], []]
        for i in member_index:
            res[0].append(self.data[0][i])
            res[1].append(self.data[1][i])

    def get_training(self):
        return self.get_data(self.tr)

    def get_testing(self):
        return self.get_data(self.te)

    def _init_elements(self):
        self.tr = []
        self.te = []
        self.re = []
