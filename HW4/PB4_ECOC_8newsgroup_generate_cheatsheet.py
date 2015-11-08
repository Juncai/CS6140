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
result_path = 'results/8newsgroupECOC_2.acc'
model_name = '8newsgroupECOC'
model_path = model_name + '.model'
threshes_path = 'data/8newsgroup/8newsgroup.thresh'
thresh_cs_path = 'data/8newsgroup/8newsgroup.thresh_cs'
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
boosts = []
function_tr_err = []
max_round = 10
cs_dict = {}

for ind, c_ecoc in enumerate(best_ecoc[1]):
    print('Generating cheat sheet {}...'.format(ind))
    # TODO preprocess the labels
    # TODO preprocess labels, so that labels match ecoc, {0, 1} -> {-1, 1}
    bin_label = copy.deepcopy(tr_data[1])
    for i in range(len(c_ecoc)):
        for j in range(len(bin_label)):
            if bin_label[j] == i:
                bin_label[j] = c_ecoc[i] if c_ecoc[i] == 1 else -1
    # TODO precompute thresholds cheat sheet
    thresh_cs = util.pre_compute_threshes_8news(tr_data[0], tr_data[1], threshes)
    cs_dict[ind] = thresh_cs
    # early terminating
    if ind == 0:
        break

with open(thresh_cs_path, 'wb+') as f:
    pickle.dump(cs_dict, f)
