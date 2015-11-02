__author__ = 'Jon'

import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b
import pickle
import math
from scipy.spatial.distance import hamming

# training parameter
result_path = 'results/8newsgroupECOC_1.acc'
model_name = '8newsgroupECOC'
model_path = model_name + '.model'
threshes_path = 'data/8newsgroup/8newsgroup.thresh'
tr_data_path = 'data/8newsgroup/train.data'
te_data_path = 'data/8newsgroup/test.data'


# laod and preprocess training data
tr_data = loader.load_pickle_file(tr_data_path)
te_data= loader.load_pickle_file(te_data_path)

# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
roc = []
auc = 0.0

tr_n = len(tr_data[0])
te_n = len(te_data[1])

# randomly generate ECOC of 20 functions
best_ecoc = [0, []]
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

# train 20 boosts
boosts = []

for c_ecoc in best_ecoc[1]:
    # TODO prepare distribution
    d = util.init_distribution(tr_n)
    boost = b.Boosting(d)
    # testing_predict = np.zeros((1, te_n)).tolist()[0]
    training_predict = np.zeros((1, tr_n)).tolist()[0]
    round_tr_err = []
    # round_te_err = []
    converged = False
    tol = 1e-5
    train_err = 2.
    round = 0
    max_round = 500
    while round < max_round:
        round += 1
        boost.add_model(ds.DecisionStump, tr_data[0], tr_data[1], threshes, c_ecoc)
        boost.update_predict(tr_data[0], training_predict)
        # boost.update_predict(te_data[0], testing_predict)
        c_model_err = boost.model[-1].n_err
        # print("Prediction 1: {}".format(testing_predict[0]))
        print("Model {} Round decision stump error: {}".format(round, c_model_err))
        c_tr_err = util.get_err_from_predict(training_predict, tr_data[1])
        print("Model {} Round training error: {}".format(round, c_tr_err))
        # round_tr_err.append(c_tr_err)
        # c_te_err = util.get_err_from_predict(testing_predict, te_data[1])
        # print("Model {} Round testing error: {}".format(round, c_te_err))
        # round_te_err.append(c_te_err)
        converged =  c_tr_err / train_err > 1 - tol
        train_err = c_tr_err
    boosts.append(boost)


# TODO calculate ecoc prediction
predict = boost.predict(te_data[0])
test_err = boost.test(te_data[0], te_data[1], util.acc)

print('Testing err is:')
print(test_err)

result = {}
result['Testingerr'] = str(test_err)

result['ROC'] = str(roc)
result['AUC'] = str(auc)

# save the model
with open(model_path, 'wb+') as f:
    pickle.dump(boost, f)

# log the training result to file
util.write_result_to_file(result_path, model_name, result)
