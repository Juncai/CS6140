import numpy as np
import math


class DecisionTree():
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
        return [self.predict(x) for x in features]

    def predict(self, feature):
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
        # find out the best feature and threshold pair if there is any
        best_pair, left_data, right_data = self.split_on_ig_v2(features, label, threshs, layer, layer_thresh)

        if best_pair[0] is None:
            # return label[0],
            return best_pair[1],
            # return np.mean(label),

        # if len(f_new_1) == 0 or len(f_new_2) == 0:
        #     return np.mean(label),
        l_tree = self.build_helper(left_data[0], left_data[1], threshs, layer+1, layer_thresh)
        r_tree = self.build_helper(right_data[0], right_data[1], threshs, layer+1, layer_thresh)
        return best_pair, (l_tree, r_tree)

    def split_on_ig_v2(self, features, label, threshs, layer, layer_thresh):
        '''
        Find the best pair based on IG
        '''
        if layer > layer_thresh:
            return (None, self.find_majority(label)), None, None

        h_y = self.compute_entropy_v3(label)
        if h_y == 0:
            return (None, self.find_majority(label)), None, None

        best_pair = None
        max_ig = 0

        for i in range(len(features[0])):
            for j in range(len(threshs[i])):
                f_cur = [x[i] for x in features]
                ig_i = self.compute_ig_v3(f_cur, label, threshs[i][j], h_y)
                if ig_i > max_ig:
                    max_ig = ig_i
                    best_pair = (i, threshs[i][j])

        if best_pair is None:
            return (None, self.find_majority(label)), None, None

        left_data, right_data = self.get_subtree_data(features, label, best_pair[0], best_pair[1])

        # in case there is not actual split happened
        if len(left_data[0]) == 0 or len(right_data[0]) == 0:
            best_pair = (None, self.find_majority(label))
        return best_pair, left_data, right_data

    # def split_on_ig(self, features, label, threshs, layer, layer_thresh):
    #     '''
    #     Find the best pair based on IG
    #     '''
    #     if layer > layer_thresh:
    #         return (None, self.find_majority(label)), None, None
    #
    #     h_y = self.compute_entropy(label)
    #     if h_y == 0:
    #         return (None, self.find_majority(label)), None, None
    #     best_pair = None
    #     max_ig = 0
    #     for i in range(len(features[0])):
    #         for j in range(len(threshs[i])):
    #             f_cur = [x[i] for x in features]
    #             ig_i = self.compute_ig(f_cur, label, threshs[i][j], h_y)
    #             if ig_i > max_ig:
    #                 max_ig = ig_i
    #                 best_pair = (i, threshs[i][j])
    #
    #     if best_pair is None:
    #         return (None, self.find_majority(label)), None, None
    #
    #     left_data, right_data = self.get_subtree_data(features, label, best_pair[0], best_pair[1])
    #
    #     # in case there is not actual split happened
    #     if len(left_data[0]) == 0 or len(right_data[0]) == 0:
    #         best_pair = (None, self.find_majority(label))
    #     return best_pair, left_data, right_data
    #
    # def compute_ig(self, feature, label, thresh, h_y):
    #     '''
    #     Compute Information Gain (Y|X)
    #     '''
    #     # compute H(Y)
    #     # h_y = self.compute_entropy(label)
    #
    #     # compute H(Y|X)
    #     f_len = len(feature)
    #     h_y_x = 0
    #     # x_1 = []
    #     y_x_1 = []
    #     # x_2 = []
    #     y_x_2 = []
    #     for i, x in enumerate(feature):
    #         if x < thresh:
    #             # x_1.append(feature[i])
    #             y_x_1.append(label[i])
    #         else:
    #             # x_2.append(feature[i])
    #             y_x_2.append(label[i])
    #     h_y_x += 1.0 * len(y_x_1) / f_len * self.compute_entropy(y_x_1)
    #     h_y_x += 1.0 * len(y_x_2) / f_len * self.compute_entropy(y_x_2)
    #     return h_y - h_y_x
    #
    # def compute_ig_v4(self, feature, label, thresh, h_y):
    #
    #     y_x_1_n = 0
    #     y_x_2_n = 0
    #     y_x_1_pos = 0
    #     y_x_1_neg = 0
    #     y_x_2_pos = 0
    #     y_x_2_neg = 0
    #
    #     for i, f in enumerate(feature):
    #         if f >= thresh:
    #             y_x_1_n += 1
    #             if label[i] == 1:
    #                 y_x_1_pos += 1
    #             else:
    #                 y_x_1_neg += 1
    #         else:
    #             y_x_2_n += 1
    #             if label[i] == 1:
    #                 y_x_2_pos += 1
    #             else:
    #                 y_x_2_neg += 1
    #     n = len(feature)
    #     h_y_x = 0
    #     h_y_x += 1.0 * y_x_1_n / n * self.compute_entropy_v4(y_x_1_n, y_x_1_neg, y_x_1_pos)
    #     h_y_x += 1.0 * y_x_2_n / n * self.compute_entropy_v4(y_x_2_n, y_x_2_neg, y_x_2_pos)
    #     return h_y - h_y_x

    def compute_ig_v3(self, feature, label, thresh, h_y):
        n = len(feature)
        h_y_x = 0
        # x_1 = []
        y_x_1 = [y for i, y in enumerate(label) if feature[i] >= thresh]
        # x_2 = []
        y_x_2 = [y for i, y in enumerate(label) if feature[i] < thresh]
        y_x_1_n = len(y_x_1)
        y_x_2_n = n - y_x_1_n
        h_y_x += 1.0 * y_x_1_n / n * self.compute_entropy_v3(y_x_1)
        h_y_x += 1.0 * y_x_2_n / n * self.compute_entropy_v3(y_x_2)
        return h_y - h_y_x

    # def compute_ig_v2(self, feature, label, thresh, h_y):
    #
    #     h_y_x = 0.
    #     n = len(label)
    #     # n_ones = np.ones((1, n))
    #     f = np.array(feature)
    #     f -= thresh
    #     f = ((np.sign(f) + 1) / 2).tolist()
    #     f_inv = [1 if ff == 0 else 0 for ff in f]
    #     # f_inv = np.logical_xor(f, n_ones)
    #     yx1_len = math.fsum(f)
    #     yx2_len = n - yx1_len
    #
    #     h_y_x += 1.0 * yx1_len / n * self.compute_entropy_v2(label, f)
    #     h_y_x += 1.0 * yx2_len / n * self.compute_entropy_v2(label, f_inv)
    #     return h_y - h_y_x
    #
    # def compute_entropy_v4(self, n, y_0_n, y_1_n):
    #     '''
    #     Compute entropy of the label
    #     '''
    #     h = 0.
    #     if y_0_n > 0:
    #         p_y_0 = y_0_n / n
    #         h += (- p_y_0 * math.log(p_y_0, 2))
    #     if y_1_n > 0:
    #         p_y_1 = y_1_n / n
    #         h += (- p_y_1 * math.log(p_y_1, 2))
    #     return h
    #
    # def compute_entropy_v2(self, label, mask=None):
    #     '''
    #     Compute entropy of the label
    #     '''
    #     n = len(label)
    #     h = 0.
    #     val_dict = {}
    #     for i, l in enumerate(label):
    #         if mask is None or mask[i] == 1:
    #             if l not in val_dict.keys():
    #                 val_dict[l] = 1
    #             else:
    #                 val_dict[l] += 1
    #     for val in val_dict.keys():
    #         p_y = 1.0 * val_dict[val] / n
    #         h += p_y * np.log2(1.0 / p_y)
    #     return h

    def compute_entropy_v3(self, label):
        '''
        Compute entropy of the label {0, 1}
        '''
        n = len(label)
        if n == 0:
            return 0
        h = 0.
        y_1_n = math.fsum(label)
        y_0_n = n - y_1_n
        if y_0_n > 0:
            p_y_0 = y_0_n / n
            h += (- p_y_0 * math.log(p_y_0, 2))
        if y_1_n > 0:
            p_y_1 = y_1_n / n
            h += (- p_y_1 * math.log(p_y_1, 2))
        return h

    # def compute_entropy(self, label):
    #     '''
    #     Compute entropy of the label
    #     '''
    #     h = 0
    #     uniq_labels = np.unique(label)
    #     label_counts = np.bincount(label)
    #     for y in uniq_labels:
    #         p_y = 1.0 * label_counts[y] / len(label)
    #         h += p_y * np.log2(1.0 / p_y)
    #     return h

    def find_majority(self, label):
        if len(label) == 1:
            return label[0]
        tmp_arr = np.array([int(y) for y in label])
        counts = np.bincount(tmp_arr)
        return np.argmax(counts)

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