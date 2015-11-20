import numpy as np
import RegressionModel as rm
import random
import Utilities as util
import math
from collections import deque


class LinearRegressionGD(rm.RegressionModel):
    iter_count = 0

    def __init__(self, theta=[]):
        self.theta = theta

    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f.tolist() for f in features]
        x = np.array(x)
        y = np.array([[l] for l in labels])

        # initialize the theta and iteration counter
        theta = np.zeros((len(x[0]), 1))
        # theta = np.array([[random.random()] for i in range(len(x[0]))])
        # theta = np.ones((len(x[0]), 1))
        self.iter_count = 0
        acc_count = [0, 0]
        # recursively update theta
        while not term_fun(theta, features, y, thresh, acc_count):
            if is_batch:
                hx = np.dot(x, theta)
                diffs = hx - y
                for i in range(len(theta)):
                    sum = 0
                    for j in range(len(diffs)):
                        sum += diffs[j][0] * x[j][i]
                    theta[i][0] = theta[i][0] - lamda * sum
            else:
                for i in range(len(x)):
                    hx = np.dot(x[i], theta)[0]
                    diff = hx - y[i][0]
                    for j in range(len(theta)):
                        theta[j][0] = theta[j][0] - lamda * diff * x[i][j]
                        if math.isnan(theta[j][0]):
                            print('something')
            self.iter_count += 1
            self.theta = theta
            # print theta


class LogisticRegressionGD(rm.RegressionModel):
    iter_count = 0
    accs = deque([])

    def __init__(self, penalty=None, alpha=None):
        '''

        :param penalty: penalty type of the regularization
        :param alpha: constant used for regularization calculation
        :return:
        '''
        self.theta = []
        self.penalty = penalty
        self.alpha = alpha

    def get_prediction(self, features):
        x = [[1] + f for f in features]
        y = self.logistic_fun_batch(self.theta, x)
        return [yy[0] for yy in y]

    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f for f in features]
        x = np.array(x)
        y = np.array([[l] for l in labels])

        f_n = len(x[0])
        n = len(features)

        # initialize the theta and iteration counter
        # theta = np.zeros((len(x[0]), 1))
        theta = np.array([[random.random()] for i in range(f_n)])

        self.iter_count = 0
        acc_count = [0, 0]

        # recursively update theta
        # if is_batch:
        #     # hx calculation
        #     hx = np.array(self.logistic_fun_batch(theta, x))
        #     while not term_fun(hx, y, thresh, acc_count):
        #         diffs = y - hx
        #         for j in range(len(theta)):
        #             sum = 0
        #             for i in range(len(diffs)):
        #                 sum += diffs[i][0] * x[i][j]
        #             theta[j][0] = theta[j][0] + lamda * sum
        # else:
        while not term_fun(theta, features, y, thresh, acc_count):
            if is_batch:
                hx = np.array(self.logistic_fun_batch(theta, x))
                diffs = y - hx
                for j in range(len(theta)):
                    if self.penalty == 'l2' and j != 0:
                        sum = 0
                        for i in range(len(diffs)):
                            sum += diffs[i][0] * x[i][j]
                        theta[j][0] = theta[j][0] + lamda * sum / n - lamda * self.alpha * theta[j][0]
                    else:
                        sum = 0
                        for i in range(len(diffs)):
                            sum += diffs[i][0] * x[i][j]
                        theta[j][0] = theta[j][0] + lamda * sum / n

            else:
                for i in range(len(x)):
                    hx = self.logistic_fun(theta, x[i])
                    diff = y[i][0] - hx
                    for j in range(len(theta)):
                        if self.penalty == 'l2':
                            if j == 0:
                                theta[j][0] = theta[j][0] + lamda * diff * x[i][j]
                            else:
                                theta[j][0] = theta[j][0] + lamda * diff * x[i][j] - lamda * self.alpha * theta[j][0]
                        else:
                            theta[j][0] = theta[j][0] + lamda * diff * x[i][j]

            self.iter_count += 1
            self.theta = theta

    def logistic_fun_batch(self, theta, features):
        '''
        Perform logistic regression calculation
        :param theta:
        :param features:
        :return:
        '''
        y = []
        for x in features:
            tmp = self.logistic_fun(theta, x)
            y.append([tmp])
        return y

    def logistic_fun(self, theta, x):
        # x = x.tolist()
        # x = [1] + x
        wx = np.dot(x, theta)[0]
        return 1.0 / (1 + np.exp(-wx))