import numpy as np


class Regression():
    theta = []

    def __init__(self, theta=[]):
        self.theta = theta

    def test(self, features, label, err_fun):
        x = [[1] + f for f in features]
        # x = features
        x = np.array(x)
        y = np.dot(x, self.theta)
        return err_fun(y, label)

    def build(self, features, label):
        '''
        Calculate theta using: theta=(XTX)-1XTY
        '''
        # add bias column
        x = [[1] + f for f in features]
        # x = features
        x = np.array(x)
        y = np.array(label)
        x_t = x.transpose()
        x_t_x = np.dot(x_t, x)
        x_t_x_i = np.linalg.pinv(x_t_x)
        x_t_x_i_x_t = np.dot(x_t_x_i, x_t)
        self.theta = np.dot(x_t_x_i_x_t, y)
