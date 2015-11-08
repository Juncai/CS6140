__author__ = 'Jon'

import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import GradientBoostedTrees as g
import copy

# training parameter
layer_thresh = 2
R = 10
result_path = 'results/housingGBT_1.err'
model_name = 'housingGBT'
threshes_path = 'data/housing_train.threshes'

# laod and preprocess training data
tr_data = loader.load_dataset('data/housing_train.txt')
te_data = loader.load_dataset('data/housing_test.txt')

# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
training_errs = []
testing_errs = []

tr_n, f_d = np.shape(tr_data[0])
round = 1
gbt = g.GradientBoostedTrees()
gbt_label = copy.deepcopy(tr_data[1])
while round <= R:
    # prepare training data
    gbt.add_tree(tr_data[0], gbt_label, threshes, layer_thresh)

    # training error is from newly added tree, testing error is from current GBT
    pred = gbt.trees[-1].batch_predict(tr_data[0])
    # training_errs.append(util.mse(pred, gbt_label))
    training_errs.append(gbt.test(tr_data[0], tr_data[1], util.mse))
    testing_errs.append(gbt.test(te_data[0], te_data[1], util.mse))

    # TODO update the labels
    gbt_label = (np.array(gbt_label) - pred).tolist()

    print('Round {} with training error: {}, testing error: {}.'.format(round, training_errs[-1], testing_errs[-1]))
    round += 1

mean_training_err = np.mean(training_errs)
mean_testing_err = np.mean(testing_errs)

# print('Training errs are:')
# print(training_errs)
# print('Mean training err is:')
# print(mean_training_err)
# print('Testing errs are:')
# print(testing_errs)
# print('Mean testing err is:')
# print(mean_testing_err)
print('Final testing err is: {}'.format(testing_errs[-1]))

result = {}
result['Trainingerrs'] = str(training_errs)
result['MeanTrainingAcc'] = str(mean_training_err)
result['Testingerrs'] = str(testing_errs)
result['MeanTestingAcc'] = str(mean_testing_err)

# log the training result to file
util.write_result_to_file(result_path, model_name, result)