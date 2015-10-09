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
                # print str(np.dot(x[0], theta)[0]) + ' ' + str(y[0][0])
                # print str(np.dot(x[-1], theta)[0]) + ' ' + str(y[-1][0])
                for i in range(len(x)):
                    hx = np.dot(x[i], theta)[0]
                    diff = hx - y[i][0]
                    for j in range(len(theta)):
                        # tmp = np.dot(x[i], theta)[0]
                        # tmp2 =labels[i][0]
                        # tmp3 = x[i][j]
                        tmp4 = theta[j][0]
                        # tmp5 = lamda * (np.dot(x[i], theta)[0] - y[i][0]) * x[i][j]
                        theta[j][0] = theta[j][0] - lamda * diff * x[i][j]
                        if math.isnan(theta[j][0]):
                            print 'something'
            self.iter_count += 1
            self.theta = theta
            # print theta


class LogisticRegressionGD(rm.RegressionModel):
    iter_count = 0
    accs = deque([])

    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f.tolist() for f in features]
        x = np.array(x)
        y = np.array([[l] for l in labels])

        # initialize the theta and iteration counter
        # theta = np.zeros((len(x[0]), 1))
        theta = np.array([[random.random()] for i in range(len(x[0]))])

        self.iter_count = 0
        acc_count = [0, 0]

        # recursively update theta
        while not term_fun(theta, features, y, thresh, acc_count):
        # while not term_fun(theta, features, y, self.accs, thresh):
            if is_batch:
                hx = np.array(util.logistic_fun_batch(theta, features))
                diffs = y - hx
                for j in range(len(theta)):
                    sum = 0
                    for i in range(len(diffs)):
                        sum += diffs[i][0] * x[i][j]
                    theta[j][0] = theta[j][0] + lamda * sum
            else:
                for i in range(len(x)):
                    hx = util.logistic_fun(theta, x[i])
                    diff = y[i][0] - hx
                    for j in range(len(theta)):
                        tmp1 = theta[j][0]
                        tmp4 = x[i][j]
                        tmp5 = theta[j][0] + lamda * diff * x[i][j]
                        theta[j][0] = theta[j][0] + lamda * diff * x[i][j]
            self.iter_count += 1
            self.theta = theta


    # def update_theta(self, ):
