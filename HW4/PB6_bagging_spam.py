__author__ = 'Jon'

import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Bagging as b

# training parameter
k = 10  # fold
layer_thresh = 2
T = 50
result_path = 'results/spamDTBagging_final.acc'
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
    tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
    tr_n, f_d = np.shape(tr_data[0])
    te_n, = np.shape(te_data[1])
    round = 0
    bagging = b.Bagging()
    while round < T:
        # prepare training data
        round += 1
        b_tr_data = util.get_bagging_data(tr_data, tr_n)
        bagging.add_tree(b_tr_data[0], b_tr_data[1], threshes, layer_thresh)
        r_tr_acc = bagging.trees[-1].test(b_tr_data[0], b_tr_data[1], util.acc)
        r_te_acc = bagging.test(te_data[0], te_data[1], util.acc)
        print('Round {} with training error: {}, testing error: {}.'.format(round, r_tr_acc, r_te_acc))

    # test the bagging model and compute testing acc
    training_errs.append(bagging.test(tr_data[0], tr_data[1], util.acc))
    testing_errs.append(bagging.test(te_data[0], te_data[1], util.acc))



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