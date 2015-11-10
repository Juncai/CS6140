
__author__ = 'Jon'

import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Bagging as b
import DecisionTree as dt
import cProfile

def main():

    target = 'v2'
    # training parameter
    k = 10  # fold
    layer_thresh = 2
    T = 50
    threshes_path = 'data/spambase.threshes'

    # laod and preprocess training data
    training_data = loader.load_dataset('data/spambase.data')

    # load thresholds
    threshes = loader.load_pickle_file(threshes_path)

    # start training
    k_folds = Preprocess.prepare_k_folds(training_data, k)
    tr_data, te_data = Preprocess.get_i_fold(k_folds, 0)
    f_cur = [x[0] for x in tr_data[0]]

    t = dt.DecisionTree()
    if target == 'v1':
        for i in range(100):
            h_y = t.compute_entropy(tr_data[1])
            thresh = threshes[0][30]
            ig = t.compute_ig(f_cur, tr_data[1], thresh, h_y)
    else:
        h_y = t.compute_entropy_v2(tr_data[1])
        thresh = threshes[0][0]
        ig = t.compute_ig_v2(f_cur, tr_data[1], thresh, h_y)

if __name__ == '__main__':
    cProfile.run('main()')
    # main()