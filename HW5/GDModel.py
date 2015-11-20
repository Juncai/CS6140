import numpy as np
import RegressionModel as rm
import random
import math
from collections import deque
import time
import Utilities as util


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

    def __init__(self, theta=None, penalty=None, alpha=None):
        '''

        :param penalty: penalty type of the regularization
        :param alpha: constant used for regularization calculation
        :return:
        '''
        self.theta = theta
        self.penalty = penalty
        self.alpha = alpha

    def get_prediction(self, features):
        x = [[1] + f for f in features]
        y = self.logistic_fun_batch(self.theta, x)
        return y

    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True, te_f=None, te_l=None):
        # construct x with the bias column
        x = [[1] + f for f in features]
        x = np.array(x)
        y = np.array(labels)

        if te_f is not None:
            te_x = np.array([[1] + f for f in te_f])
            te_y = np.array(te_l)
            te_n = len(te_f)

        f_n = len(x[0])
        n = len(features)

        # initialize the theta and iteration counter
        # theta = np.zeros((len(x[0]), 1))
        if self.theta is None:
            # theta = np.array([random.random() for i in range(f_n)])
            theta = np.array([0.1 for i in range(f_n)])
        else:
            theta = self.theta

        self.iter_count = 0
        acc_count = [0, 0]      # [cur_acc, counter]
        done = False
        hx = np.array(self.logistic_fun_batch(theta, x))

        while not done:
        # while not term_fun(theta, te_x, te_y, thresh, acc_count):
            ts = time.time()
            if is_batch:
                diffs = y - hx
                for j in range(len(theta)):
                    if self.penalty == 'l2' and j != 0:
                        x_j = x[:,j]
                        sum = np.dot(diffs, x_j)
                        theta[j] = theta[j] + lamda * sum / n - lamda * self.alpha * theta[j] / n
                    else:
                        x_j = x[:,j]
                        sum = np.dot(diffs, x_j)
                        theta[j] = theta[j] + lamda * sum / n

            else:
                for i in range(len(x)):
                    hx = self.logistic_fun(theta, x[i])
                    diff = y[i] - hx
                    for j in range(len(theta)):
                        if self.penalty == 'l2':
                            if j == 0:
                                theta[j] = theta[j] + lamda * diff * x[i][j] / n
                            else:
                                theta[j] = theta[j] + lamda * diff * x[i][j] / n - lamda * self.alpha * theta[j] / n
                        else:
                            theta[j] = theta[j] + lamda * diff * x[i][j] / n

            self.iter_count += 1
            self.theta = theta

            # check the termination condition
            hx = np.array(self.logistic_fun_batch(theta, x))
            tr_acc = (np.round(hx) == y).mean()
            tr_mse = ((hx - y) ** 2).mean()
            if acc_count[0] == tr_acc:
                acc_count[1] += 1
            else:
                acc_count[0] = tr_acc
                acc_count[1] = 1
            # testing results
            te_hx = np.array(self.logistic_fun_batch(theta, te_x))
            te_acc = (np.round(te_hx) == te_y).mean()
            te_mse = ((te_hx - te_y) ** 2).mean()


            print('Training acc: {}, mse: {}, testing acc: {}, testing mse: {}, time used: {}'.format(tr_acc, tr_mse, te_acc, te_mse, time.time() - ts))
            if te_acc >= thresh and tr_acc >= thresh:
                done = True


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
            y.append(tmp)
        return y

    def logistic_fun(self, theta, x):
        # x = x.tolist()
        # x = [1] + x
        wx = np.dot(x, theta)
        return 1.0 / (1 + np.exp(-wx))