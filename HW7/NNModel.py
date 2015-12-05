import numpy as np
import random
import Utilities as util

class Perceptron():
    training_record = {}
    theta = []

    def __init__(self, theta=[]):
        self.theta = theta

    def build(self, features, labels, lamda, term_fun, thresh, is_batch=True):
        # construct x with the bias column
        x = [[1] + f for f in features]
        for i in range(len(x)):
            if labels[i][0] == -1:
                x[i] = [-xx for xx in x[i]]
                labels[i] = 1

        x = np.array(x)
# mistakes list
        m = []
        # init training records
        self.training_record.clear()

        # initialize the theta and iteration counter
        # self.theta = np.zeros((len(x[0]), 1))
        self.theta = [[random.random()] for i in range(len(x[0]))]
        iter_count = 1

        # recursively update theta
        while True:
            # m = filter(self.filter_mistakes, x)
            m = [xx for xx in x if np.dot(xx, self.theta) <= 0]
            self.training_record[iter_count] = len(m)
            print('Iteration {}, total_mistake {}.'.format(iter_count, len(m)))
            if term_fun(m, thresh):  # terminating condition
                break
            if is_batch:
                for x_i in m:
                    self.theta = np.add(self.theta, np.multiply(lamda, np.transpose([x_i], (1, 0))))
            iter_count += 1  # count the iterations

    def filter_mistakes(self, x):
        return np.dot(x, self.theta) <= 0


class NN():
    w_in = []  # weights between input layer and hidden layer,  3 * 9
    w_out = []  # weights between hidden layer and output layer,  8 * 4
    y = []  # outputs of hidden layer
    z = []  # outputs of output layer
    iter_count = 0
    mse = 0
    def __init__(self):
        pass

    def build(self, data, lamda, term_fun, thresh):
        # init input and output
        nn_input = [[1] + x for x in data[0]]
        target = data[1]

        # init weights
        w_in = [[random.random() for i in range(9)] for j in range(3)]
        w_out = [[random.random() for i in range(4)] for j in range(8)]
        print('Initial weights: {}, {}'.format(w_in, w_out))
        nn_output = [[], []]
        # train the network
        while True:
            self.iter_count += 1
            nn_output = self.nn_output(nn_input, w_in, w_out)
            self.y = nn_output[1]
            self.z = nn_output[0]
            self.w_in = w_in
            self.w_out = w_out
            # print nn_output
            # terminate when error satisfied
            # self.mse = term_fun(nn_output[0], target)
            if self.iter_count > thresh:
                break
            for data_i in range(len(nn_input)):

                # compute hidden layer outputs
                y = self.layer_output(nn_input[data_i], w_in)
                y_w_bias = [1] + y

                # compute output layer outputs
                z = self.layer_output(y_w_bias, w_out)

                # TODO update weights close to the output layer
                err_z = []
                for i in range(len(target[data_i])):
                    err_z.append((target[data_i][i] - z[i]) * z[i] * (1 - z[i]))
                for i in range(len(w_out[0])):
                    for j in range(len(w_out)):
                        w_out[j][i] = w_out[j][i] + lamda * err_z[j] * y_w_bias[i]

                # TODO update weights close to the input layer
                err_y = []
                for i in range(len(y)):
                    sum = 0
                    for j in range(len(z)):
                        sum += err_z[j] * w_out[j][i + 1]
                    err_y.append(sum * y[i] * (1 - y[i]))
                for i in range(len(w_in[0])):
                    for j in range(len(w_in)):
                        w_in[j][i] = w_in[j][i] + lamda * err_y[j] * nn_input[data_i][i]



    def layer_output(self, input, weights):
        '''
        Compute layer output from given input and weights
        :param input: 1 * 9
        :param weights:
        :return:
        '''
        net = np.transpose(np.dot(weights, np.transpose([input], (1, 0))), (1, 0))[0]
        return [util.sigmoid(x) for x in net]


    def nn_output(self, inputs, w_in, w_out):
        net_y = np.dot(inputs, np.transpose(w_in, (1, 0)))  # 8 * 3
        out_y = map(lambda y: map(util.sigmoid, y), net_y)  # 8 * 3
        in_z = [[1] + y for y in out_y]  # 8 * 4
        net_z = np.dot(in_z, np.transpose(w_out, (1, 0)))  # 8 * 8
        output = map(lambda z: map(util.sigmoid, z), net_z)  # 8 * 8
        return output, out_y


