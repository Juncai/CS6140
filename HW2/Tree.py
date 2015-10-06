import numpy as np
import Utilities as util


class Tree():
    tree = ()

    def __init__(self, tree=()):
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



class DecisionTree(Tree):
    acc = 0

    def test(self, features, label):
        '''
        Take a tree model and a test dataset,
        Return the result
        '''
        predict = self.batch_predict(features)
        return util.compute_acc_confusion_matrix(predict, label)

    def build(self, features, label, threshs, term_con, term_thresh):
        self.tree = self.build_helper(features, label, threshs, 1, term_con, term_thresh)

    def build_helper(self, features, label, threshs, layer, term_con, term_thresh):
        '''
        Build a DT with given dataset, criteria function and thresholds
        return a tree as a tuple
        '''
        # terminating case
        # cri = term_fun(features, label, layer, add)
        # if cri[0]:
        #     return cri[1],

        # find out the best feature and threshold pair if there is any
        best_pair, left_data, right_data = self.split_on_ig(features, label, threshs, layer, term_con, term_thresh)

        if best_pair[0] is None:
            # return label[0],
            return best_pair[1],
            # return np.mean(label),

        # if len(f_new_1) == 0 or len(f_new_2) == 0:
        #     return np.mean(label),
        l_tree = self.build_helper(left_data[0], left_data[1], threshs, layer+1, term_con, term_thresh)
        r_tree = self.build_helper(right_data[0], right_data[1], threshs, layer+1, term_con, term_thresh)
        return best_pair, (l_tree, r_tree)


    def split_on_ig(self, features, label, threshs, layer, term_con, term_thresh):
        '''
        Find the best pair based on IG
        '''
        best_pair = None
        if layer > term_thresh or util.compute_entropy(label) == 0:
            return (None, util.find_majority(label)), None, None
        max_ig = 0
        for i in range(len(features[0])):
            for j in range(len(threshs[i])):
                f_cur = [x[i] for x in features]
                ig_i = util.compute_ig(f_cur, label, threshs[i][j])
                if ig_i > max_ig:
                    max_ig = ig_i
                    best_pair = (i, threshs[i][j])

        if best_pair is None:
            return (None, util.find_majority(label)), None, None

        left_data, right_data = util.get_subtree_data(features, label, best_pair[0], best_pair[1])

        # in case there is not actual split happened
        if len(left_data[0]) == 0 or len(right_data[0]) == 0:
            best_pair = (None, util.find_majority(label))
        return best_pair, left_data, right_data