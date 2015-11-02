import numpy as np
import math


class RegressionTree():
    tree = []

    def __init__(self, tree=[]):
        self.tree = tree

    def test(self, features, label, err_fun):
        '''
        Take a tree model and a test dataset,
        Return the result
        '''
        predict = self.batch_predict(features)
        return err_fun(predict, label)

    def batch_predict(self, features):
        '''
        Comsume a tree model and datapoints
        Return the predictions
        '''
        return [self.single_predict(x) for x in features]

    def single_predict(self, feature):
        '''
        Comsume a tree model and a single datapoint
        Return the prediction
        '''
        c_node = self.tree
        while len(c_node) > 1:
            if feature[c_node[0][0]] < c_node[0][1]:
                c_node = c_node[1][0]
            else:
                c_node = c_node[1][1]
        return c_node[0]

    def build(self, features, label, threshs, layer_thresh):
        self.tree = self.build_helper(features, label, threshs, 1, layer_thresh)

    def build_helper(self, features, label, threshs, layer, layer_thresh):
        '''
        Build a DT with given dataset, criteria function and thresholds
        return a tree as a tuple
        '''
        # terminating case
        # cri = term_fun(features, label, layer, add)
        # if cri[0]:
        #     return cri[1],

        # find out the best feature and threshold pair if there is any
        best_pair, left_data, right_data = self.split_on_mse(features, label, threshs, layer, layer_thresh)

        if best_pair[0] is None:
            # return label[0],
            return best_pair[1],
            # return np.mean(label),

        # if len(f_new_1) == 0 or len(f_new_2) == 0:
        #     return np.mean(label),
        l_tree = self.build_helper(left_data[0], left_data[1], threshs, layer+1, layer_thresh)
        r_tree = self.build_helper(right_data[0], right_data[1], threshs, layer+1, layer_thresh)
        return best_pair, (l_tree, r_tree)

    def split_on_mse(self, features, label, threshs, layer, term_thresh):
        '''
        Find the best pair based on mse
        '''
        # check the terminating condition
        if layer > term_thresh:
            return (None, np.mean(label)), None, None

        best_pair = None
        cur_err = self.reg_error(label, np.mean(label))
        min_err = float('inf')
        for i in range(len(features[0])):
            for j in range(len(threshs[i])):
                f_cur = [x[i] for x in features]
                err_i = self.err_all(f_cur, label, threshs[i][j])
                if err_i < min_err:
                    min_err = err_i
                    best_pair = (i, threshs[i][j])

        left_data, right_data = self.get_subtree_data(features, label, best_pair[0], best_pair[1])

        # in case there is not actual split happened
        if len(left_data[0]) == 0 or len(right_data[0]) == 0:
            best_pair = (None, np.mean(label))
        return best_pair, left_data, right_data

    def reg_error(self, labels, predict):
        if len(labels) == 0:
            return 0
        res = 0.0
        for l in labels:
            res += math.pow(predict - l, 2)
        return res

    def err_all(self, feature, label, thresh):
        res = 0
        y_x_1 = []
        y_x_2 = []
        for i, x in enumerate(feature):
            if x < thresh:
                y_x_1.append(label[i])
            else:
                y_x_2.append(label[i])
        if len(y_x_1):
            y_x_1_mean = np.mean(y_x_1)
            # y_pre_1 = [y_x_1_mean for j in range(len(y_x_1))]
            res += self.reg_error(y_x_1, y_x_1_mean)
        if len(y_x_2):
            y_x_2_mean = np.mean(y_x_2)
            # y_pre_2 = [y_x_2_mean for j in range(len(y_x_2))]
            res += self.reg_error(y_x_2, y_x_2_mean)
        return  res

    def get_subtree_data(self, features, label, f_index, thresh):
        # generate features and label for subtrees
        f_new_1 = []
        l_new_1 = []
        f_new_2 = []
        l_new_2 = []
        for i in range(len(label)):
            if features[i][f_index] < thresh:
                f_new_1.append(features[i])
                l_new_1.append(label[i])
            else:
                f_new_2.append(features[i])
                l_new_2.append(label[i])
        return ((f_new_1, l_new_1), (f_new_2, l_new_2))