import numpy as np
import RegressionModel as rm


class LinearRegressionGD(rm.RegressionModel):
    iter_count = 0

    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f for f in features]
        x = np.array(x)
        y = [[l] for l in labels]

        # initialize the theta and iteration counter
        theta = np.zeros((len(x[0]), 1))
        self.iter_count = 0

        # recursively update theta
        while not term_fun(theta, features, y, thresh):
            if is_batch:
                hx = np.dot(x, theta)
                diffs = hx - y
                for i in range(len(theta)):
                    sum = 0
                    for j in range(len(diffs)):
                        sum += diffs[j][0] * x[j][i]
                    theta[i][0] = theta[i][0] - lamda * sum
            else:
                for i in range(len(theta)):
                    for j in range(len(features[0])):
                        theta[i] = theta[i] - lamda * (np.dot(x[j], theta) - labels[j]) * x[j]
            self.iter_count += 1

        self.theta = theta


class LogisticRegressionGD(rm.RegressionModel):
    iter_count = 0

    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f for f in features]
        x = np.array(x)
        y = [[l] for l in labels]

        # initialize the theta and iteration counter
        theta = np.zeros((len(x[0]), 1))
        self.iter_count = 0

        # recursively update theta
        while not term_fun(theta, features, y, thresh):
            if is_batch:
                hx = np.array(self.logistic_fun(theta, features))
                diffs = y - hx
                for i in range(len(theta)):
                    sum = 0
                    for j in range(len(diffs)):
                        sum += diffs[j][0] * x[j][i]
                    theta[i][0] = theta[i][0] + lamda * sum
            else:
                for i in range(len(theta)):
                    for j in range(len(features[0])):
                        # TODO refine the update of the theta
                        theta[i] = theta[i] - lamda * (np.dot(x[j], theta) - labels[j]) * x[j]
            self.iter_count += 1

        self.theta = theta

    def logistic_fun(self, theta, features):
        '''
        Perform logistic regression calculation
        :param theta:
        :param features:
        :return:
        '''
        y = []
        for x in features:
            x = [1] + x
            tmp = np.dot(x, theta)
            y.append([1.0 / (1 + np.exp(-tmp))])
        return y


class Perceptron(rm.RegressionModel):
    training_record = {}
    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f for f in features]
        x = np.array(x)
        y = [[l] for l in labels]
        # mistakes list
        m = []
        #

        # initialize the theta and iteration counter
        theta = np.zeros((len(x[0]), 1))
        self.iter_count = 0

        # recursively update theta
        while not term_fun(theta, features, y, thresh):
            if is_batch:
                hx = np.dot(x, theta)
                diffs = hx - y
                for i in range(len(theta)):
                    sum = 0
                    for j in range(len(diffs)):
                        sum += diffs[j][0] * x[j][i]
                    theta[i][0] = theta[i][0] - lamda * sum
            else:
                for i in range(len(theta)):
                    for j in range(len(features[0])):
                        theta[i] = theta[i] - lamda * (np.dot(x[j], theta) - labels[j]) * x[j]
            self.iter_count += 1
