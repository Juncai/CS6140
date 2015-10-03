import numpy as np


class RegressionModel():
    theta = []

    def test(self, features, label, err_fun):
        x = [[1] + f for f in features]
        # x = features
        x = np.array(x)
        y = np.dot(x, self.theta)
        y_1_d = [yy[0] for yy in y]
        return err_fun(y_1_d, label)


class LinearRegression(RegressionModel):
    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, label):
        '''
        Calculate theta using: theta=(XTX)-1XTY
        '''
        # add bias column
        x = [[1] + f for f in features]
        # x = features
        x = np.array(x)
        # y = np.array(label)
        y = np.array([[l] for l in label])
        x_t = x.transpose()
        x_t_x = np.dot(x_t, x)
        x_t_x_i = np.linalg.pinv(x_t_x)
        x_t_x_i_x_t = np.dot(x_t_x_i, x_t)
        self.theta = np.dot(x_t_x_i_x_t, y)

class Ridge(RegressionModel):
    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, label, lamda=0.5):
        '''
        Calculate theta using: theta=(XTX+lambdaI)-1XTY
        '''
        # add bias column
        x = [[1] + f for f in features]
        # construct lambdaI
        lambda_i = lamda * np.eye(len(features[0]))

        x = np.array(x)
        y = np.array(label)
        x_t = x.transpose()
        x_t_x = np.dot(x_t, x)
        x_t_x_lambda_i = x_t_x + lambda_i
        x_t_x_lambda_i_inv = np.linalg.pinv(x_t_x_lambda_i)
        x_t_x_lambda_i_inv_x_t = np.dot(x_t_x_lambda_i_inv, x_t)
        self.theta = np.dot(x_t_x_lambda_i_inv_x_t, y)