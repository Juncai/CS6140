import numpy as np
import Utilities as util


class RegressionModel():
    theta = []

    def test(self, features, label, err_fun):
        y_1_d = self.get_prediction(features)
        return err_fun(y_1_d, label)

    def calculate_roc(self, features, label):
        y_1_d = self.get_prediction(features)
        print(y_1_d)
        d = []
        for i in range(len(label)):
            d.append([y_1_d[i], label[i]])
        d.sort(key=lambda x: x[0],reverse=True)
        d_predict = [y[0] for y in d]
        d_label = [y[1] for y in d]
        pos = 0
        neg = 0
        for dl in d_label:
            if dl == 1:
                pos += 1
            else:
                neg += 1
        roc = []
        for i in range(len(d)):
            roc.append(self.false_pos_true_pos(d_predict, d_label,
                                               pos, neg, i))
        return roc


    def false_pos_true_pos(self, pred, label, pos, neg, ind):
        false_pos = 0
        true_pos = 0
        for i in range(ind+1):
            if label[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        return (float(false_pos) / neg if neg > 0 else 0,
                float(true_pos) / pos if pos > 0 else 1)


    def get_prediction(self, features):
        x = [[1] + f for f in features]
        x = np.array(x)
        y = np.dot(x, self.theta)
        y_1_d = [yy[0] for yy in y]
        return y_1_d


class LinearRegression(RegressionModel):
    def __init__(self, theta=[]):
        self.theta = theta


    def build(self, features, label):
        '''
        Calculate theta using: theta=(XTX)-1XTY
        '''
        # add bias column
        x = [[1] + f.tolist() for f in features]
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
        x = [[1] + f.tolist() for f in features]
        # construct lambdaI
        lambda_i = lamda * np.eye(len(x[0]))

        x = np.array(x)
        y = np.array([[l] for l in label])
        x_t = np.transpose(x, (1, 0))
        x_t_x = np.dot(x_t, x)
        x_t_x_lambda_i = x_t_x + lambda_i
        x_t_x_lambda_i_inv = np.linalg.pinv(x_t_x_lambda_i)
        x_t_x_lambda_i_inv_x_t = np.dot(x_t_x_lambda_i_inv, x_t)
        self.theta = np.dot(x_t_x_lambda_i_inv_x_t, y)