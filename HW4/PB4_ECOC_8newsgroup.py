__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b
import pickle
import math
from scipy.spatial.distance import hamming
import os.path as path
import copy

# training parameter
result_path = 'results/8newsgroupECOC_1.acc'
model_name = '8newsgroupECOC'
model_path = model_name + '.model'
threshes_path = 'data/8newsgroup/8newsgroup.thresh'
tr_data_path = 'data/8newsgroup/train.data'
te_data_path = 'data/8newsgroup/test.data'
ecoc_path = 'data/8newsgroup/ecoc'


# laod and preprocess training data
tr_data = loader.load_pickle_file(tr_data_path)
te_data= loader.load_pickle_file(te_data_path)

# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
tr_n = len(tr_data[0])
te_n = len(te_data[1])

# randomly generate ECOC of 20 functions
if path.isfile(ecoc_path):
    print('Loading the ecoc...')
    best_ecoc = loader.load_pickle_file(ecoc_path)
else:
    print('Creating the ecoc...')
    best_ecoc = [0, [], []]     # distance, ecoc for training, ecoc for predicting
    for i in range(100):
        n = int(math.pow(2, 20))
        codes = np.random.choice(n, 8, replace=False)
        ecoc_func_codes = []
        for i in range(20):
            ecoc_func_codes.append([])
        c_ecoc = []
        for c in codes:
            bin_s = '{0:020b}'.format(c)
            bin_s = [int(ss) for ss in bin_s]
            c_ecoc.append(bin_s)
            for i in range(20):
                ecoc_func_codes[i].append(bin_s[i])
        c_hamming_dist = 0
        for j in range(len(c_ecoc)):
            for k in range(len(c_ecoc)):
                if j != k:
                    c_hamming_dist += hamming(c_ecoc[j], c_ecoc[k])
        if c_hamming_dist > best_ecoc[0]:
            best_ecoc[0] = c_hamming_dist
            best_ecoc[1] = ecoc_func_codes
            best_ecoc[2] = c_ecoc

    # serialize the best ecoc
    with open(ecoc_path, 'wb+') as f:
        pickle.dump(best_ecoc, f)

print('Init ecoc done!')

# train 20 boosts
print('Begin training...')
boosts = []
function_tr_err = []
max_round = 10

for ind, c_ecoc in enumerate(best_ecoc[1]):
    print('Training function {}...'.format(ind))
    # TODO preprocess the labels
    # TODO preprocess labels, so that labels match ecoc, {0, 1} -> {-1, 1}
    bin_label = copy.deepcopy(tr_data[1])
    for i in range(len(c_ecoc)):
        for j in range(len(bin_label)):
            if bin_label[j] == i:
                bin_label[j] = c_ecoc[i] if c_ecoc[i] == 1 else -1
    # TODO prepare distribution
    d = util.init_distribution(tr_n)
    boost = b.Boosting(d)
    # testing_predict = np.zeros((1, te_n)).tolist()[0]
    training_predict = np.zeros((1, tr_n)).tolist()[0]
    round_tr_err = []
    # round_te_err = []
    # converged = False
    # tol = 1e-5
    # train_err = 2.
    round = 0
    while round < max_round:
        round += 1
        boost.add_model(ds.DecisionStump, tr_data[0], bin_label, threshes, c_ecoc)
        boost.update_predict(tr_data[0], training_predict)
        # boost.update_predict(te_data[0], testing_predict)
        c_model_err = boost.model[-1].n_err
        # print("Prediction 1: {}".format(testing_predict[0]))
        print("Model {} Round decision stump error: {}".format(round, c_model_err))
        c_tr_err = util.get_err_from_predict(training_predict, bin_label)
        print("Model {} Round training error: {}".format(round, c_tr_err))
        round_tr_err.append(c_tr_err)
        # c_te_err = util.get_err_from_predict(testing_predict, te_data[1])
        # print("Model {} Round testing error: {}".format(round, c_te_err))
        # round_te_err.append(c_te_err)
        # converged =  c_tr_err / train_err > 1 - tol
        # train_err = c_tr_err
    function_tr_err.append(boost.test(tr_data[0], bin_label, util.acc))
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
result['Testingerr'] = str(test_err)
result['Trainingerr'] = str(train_err)
result['RoundTrainingData'] = str(function_tr_err)

# save the model
with open(model_path, 'wb+') as f:
    pickle.dump(boosts, f)

# log the training result to file
util.write_result_to_file(result_path, model_name, result)
