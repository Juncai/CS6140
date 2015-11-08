import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b

# training parameter
# target = 'crx'
target = 'vote'
k = 10  # fold
round_limit = 100
if target == 'crx':
    result_path = 'results/crxBoosting_cPercent_1.acc'
    model_name = 'crx_' + str(k) + 'fold'
    threshes_path = 'data/crx.threshes'
    data_path = 'data/crx_parsed.data'
else:
    result_path = 'results/voteBoosting_1.acc'
    model_name = 'vote_' + str(k) + 'fold'
    threshes_path = 'data/vote.threshes'
    data_path = 'data/vote_parsed.data'

# laod and preprocess training data
training_data = loader.load_pickle_file(data_path)
print('total data points: {}'.format(len(training_data[0])))
# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
training_errs_by_percent = {}
testing_errs_by_percent = {}
auc_by_percent = {}
roc = []
auc = 0.0
k_folds = Preprocess.prepare_k_folds(training_data, k)

for i in range(k):
    tr_data_all, te_data = Preprocess.get_i_fold(k_folds, i)

    for c in (5, 10, 15, 20, 30, 50, 80):
        if c not in training_errs_by_percent.keys():
            training_errs_by_percent[c] = []
            testing_errs_by_percent[c] = []
            auc_by_percent[c] = []

        tr_data = Preprocess.get_c_percent(c, tr_data_all)

        tr_n, f_d = np.shape(tr_data[0])
        te_n, = np.shape(te_data[1])
        # TODO prepare distribution
        d = util.init_distribution(len(tr_data[0]))
        # TODO compute thresholds cheat sheet
        thresh_cs = util.pre_compute_threshes_uci(tr_data[0], tr_data[1], threshes)
        boost = b.Boosting(d)
        testing_predict = np.zeros((1, te_n)).tolist()[0]
        training_predict = np.zeros((1, tr_n)).tolist()[0]
        round_tr_err = []
        round_te_err = []
        round_model_err = []
        round_te_auc = []
        converged = False
        tol = 1e-5
        te_auc = 2.
        round = 0
        while round < round_limit: # and not converged:
            round += 1
            boost.add_model(ds.DecisionStump, tr_data[0], tr_data[1], threshes, thresh_cs)
            boost.update_predict(tr_data[0], training_predict)
            boost.update_predict(te_data[0], testing_predict)
            c_model_err = boost.model[-1].w_err
            round_model_err.append(c_model_err)
            c_f_ind = boost.model[-1].f_ind
            c_thresh = boost.model[-1].thresh
            c_tr_err = util.get_err_from_predict(training_predict, tr_data[1])
            c_te_err = util.get_err_from_predict(testing_predict, te_data[1])
            # TODO calculate the AUC for testing results
            c_te_auc = util.get_auc_from_predict(testing_predict, te_data[1])
            # round_tr_err.append(c_tr_err)
            # round_te_err.append(c_te_err)
            # round_te_auc.append(c_te_auc)
            print('{} % data Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {:.12f} AUC: {:.12f}'.format(c, round, c_f_ind, c_thresh, c_model_err, c_tr_err, c_te_err, c_te_auc))
            print('Num of training data: {}'.format(tr_n))
            # converged =  abs(c_te_auc - te_auc) / te_auc <= tol
            # te_auc = c_te_auc

            # for test
            # if c == 5 and c_tr_err < 0.04:
            #     break
        training_errs_by_percent[c].append(c_tr_err)
        testing_errs_by_percent[c].append(c_te_err)
        auc_by_percent[c].append(c_te_auc)


mean_training_err = {}
mean_testing_err = {}
for c_k in training_errs_by_percent.keys():
    mean_training_err[c_k] = np.mean(training_errs_by_percent[c_k])
    mean_testing_err[c_k] = np.mean(testing_errs_by_percent[c_k])

print(str(k) + '-fold validation done. Training errs are:')
print(training_errs_by_percent)
print('Mean training err is:')
print(mean_training_err)
print('Testing errs are:')
print(testing_errs_by_percent)
print('Mean testing err is:')
print(mean_testing_err)
print('Testing auc are:')
print(auc_by_percent)

result = {}
result['Fold'] = str(k)
result['Trainingerrs'] = str(training_errs_by_percent)
result['MeanTrainingAcc'] = str(mean_training_err)
result['Testingerrs'] = str(testing_errs_by_percent)
result['MeanTestingAcc'] = str(mean_testing_err)
result['AUC'] = str(auc_by_percent)

# result['1stBoostTrainingError'] = str(tr_errs_1st_boost)
# result['1stBoostTestingError'] = str(te_errs_1st_boost)
# result['1stBoostModelError'] = str(round_err_1st_boost)
# result['1stBoostTestingAUC'] = str(te_auc_1st_boost)

# result['ROC'] = str(roc)



# log the training result to file
util.write_result_to_file(result_path, model_name, result)