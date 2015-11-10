__author__ = 'Jon'

import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Bagging as b
import DecisionTree as dt
import cProfile
import time

def main():
    # training parameter
    k = 10  # fold
    layer_thresh = 2
    T = 50
    result_path = 'results/spamDT_final.acc'
    model_name = 'spam_' + str(k) + 'fold'
    threshes_path = 'data/spambase.threshes'

    # laod and preprocess training data
    training_data = loader.load_dataset('data/spambase.data')

    # load thresholds
    threshes = loader.load_pickle_file(threshes_path)

    # start training
    training_errs = []
    testing_errs = []
    roc = []
    auc = 0.0
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in range(1):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
        tr_n, f_d = np.shape(tr_data[0])
        te_n, = np.shape(te_data[1])
        t = dt.DecisionTree()
        t.build(tr_data[0], tr_data[1], threshes, layer_thresh)
        # test the bagging model and compute testing acc
        training_errs.append(t.test(tr_data[0], tr_data[1], util.acc))
        testing_errs.append(t.test(te_data[0], te_data[1], util.acc))
        print('Round {} finishes, time used: {}'.format(i, time.time() - st))


    mean_training_err = np.mean(training_errs)
    mean_testing_err = np.mean(testing_errs)

    print(str(k) + '-fold validation done. Training errs are:')
    print(training_errs)
    print('Mean training err is:')
    print(mean_training_err)
    print('Testing errs are:')
    print(testing_errs)
    print('Mean testing err is:')
    print(mean_testing_err)

    result = {}
    result['Fold'] = k
    result['Trainingerrs'] = training_errs
    result['MeanTrainingAcc'] = mean_training_err
    result['Testingerrs'] = testing_errs
    result['MeanTestingAcc'] = mean_testing_err

    result['ROC'] = roc
    result['AUC'] = auc



    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)


if __name__ == '__main__':
    cProfile.run('main()')
    # main()