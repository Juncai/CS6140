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
import time

# training parameter
result_path = 'results/8newsgroupECOC_3.acc'
model_name = '8newsgroupECOC_cs'
model_path = 'results/8newsgroup/' + model_name + '.model'
model2_path = 'results/8newsgroup/8newsgroupECOC_cs_2.model'
threshes_path = 'data/8newsgroup/8newsgroup.thresh'
tr_data_path = 'data/8newsgroup/train.data'
te_data_path = 'data/8newsgroup/test.data'
ecoc_path = 'data/8newsgroup/ecoc_cs'

print('Loading boosts...')
boosts = loader.load_pickle_file(model_path)
boosts2 = loader.load_pickle_file(model2_path)





print('Loading the ecoc...')
best_ecoc = loader.load_pickle_file(ecoc_path)


# laod and preprocess training data
tr_data = loader.load_pickle_file(tr_data_path)
te_data= loader.load_pickle_file(te_data_path)


# start training
# tr_n = len(tr_data[0])
# te_n = len(te_data[1])
#
# # TODO calculate ecoc prediction
# # training error
train_err = util.ecoc_test(tr_data[0], tr_data[1], boosts, best_ecoc[2])
# test_err = util.ecoc_test(te_data[0], te_data[1], boosts, best_ecoc[2])
# test_err2 = util.ecoc_test(te_data[0], te_data[1], boosts2, best_ecoc[2])
#
print('Training err is: {}'.format(train_err))
# print('Testing err is: {}'.format(test_err))
# print('Testing err 2 is: {}'.format(test_err2))
# print(best_ecoc[0])
# print(best_ecoc[2])

