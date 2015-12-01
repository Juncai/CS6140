__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import Utilities as util
import pickle
import math
from scipy.spatial.distance import hamming
import os.path as path
import copy
import MySVM as svm
import time

wl_type = ''

# training parameter
result_path = 'results/digits_ECOC_' + wl_type + '_1.acc'
model_name = 'digits_' + wl_type + '_1'
model_path = 'results/' + model_name + '.model'
tr_data_path = 'data\\digits\\tr_f_l.pickle'
te_data_path = 'data\\digits\\te_f_l.pickle'
threshes_path = 'data\\digits\\sel_tr.threshes'


# laod and preprocess training data
tr_data = loader.load_pickle_file(tr_data_path)
te_data= loader.load_pickle_file(te_data_path)

# transpose label
tr_data[1] = np.transpose(tr_data[1])[0]
te_data[1] = np.transpose(te_data[1])[0]

# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
tr_n = len(tr_data[0])
te_n = len(te_data[1])


# train 50 boosts
print('Begin training...')
boosts = []
function_tr_err = []

max_round = 200
if wl_type == 'random_':
    max_round = 2000

for ind, c_ecoc in enumerate(best_ecoc[1]):
    print('Training function {}...'.format(ind))
    # TODO preprocess labels, so that labels match ecoc, {0, 1} -> {-1, 1}
    bin_label = util.generate_bin_label_from_ecoc(tr_data[1], c_ecoc)

    # TODO prepare distribution
    d = util.init_distribution(tr_n)
    thresh_cs = None
    if wl == ds.DecisionStump:
        # TODO precompute thresholds cheat sheet
        thresh_cs = util.pre_compute_threshes(tr_data[0], bin_label, threshes)
    boost = b.Boosting(d)
    training_predict = np.zeros((1, tr_n)).tolist()[0]
    round_tr_err = []
    round = 0
    while round < max_round:
        st = time.time() # start ts
        round += 1
        boost.add_model(wl, tr_data[0], bin_label, threshes, thresh_cs)
        boost.update_predict(tr_data[0], training_predict)
        c_model_err = boost.model[-1].w_err
        c_tr_err = util.get_err_from_predict(training_predict, bin_label)
        round_tr_err.append(c_tr_err)
        c_f_ind = boost.model[-1].f_ind
        c_thresh = boost.model[-1].thresh
        print('Time used: {}'.format(time.time() - st))
        print('Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {} AUC {}'.format(round, c_f_ind, c_thresh, c_model_err, c_tr_err, 0, 0))
        if c_tr_err == 0:
            break
    function_tr_err.append(c_tr_err)
    boosts.append(boost)

print('Training done.')

# TODO calculate ecoc prediction
# training error
train_err = util.ecoc_test(tr_data[0], tr_data[1], boosts, best_ecoc[2])
test_err = util.ecoc_test(te_data[0], te_data[1], boosts, best_ecoc[2])

print('Training err is: {}'.format(train_err))
print('Testing err is: {}'.format(test_err))
print('Training err for each function: ')
print(str(function_tr_err))

result = {}
result['Testingerr'] = test_err
result['Trainingerr'] = train_err
result['RoundTrainingData'] = function_tr_err
result['ECOC'] = best_ecoc[2]

# save the model
with open(model_path, 'wb+') as f:
    pickle.dump(boosts, f)

# log the training result to file
util.write_result_to_file(result_path, model_name, result, True)
