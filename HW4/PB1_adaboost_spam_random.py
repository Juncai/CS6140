import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b

# training parameter
k = 10  # fold
result_path = 'results/spamRDSBoosting_1.acc'
model_name = 'spam_' + str(k) + 'fold'
threshes_path = 'data/spambase.threshes'

# laod and preprocess training data
training_data = loader.load_dataset('data/spambase.data')
# TODO convert labels from {0, 1} to {-1, 1}
util.replace_zero_label_with_neg_one(training_data)

# load thresholds
threshes = loader.load_pickle_file(threshes_path)

# start training
training_errs = []
testing_errs = []
roc = []
auc = 0.0
k_folds = Preprocess.prepare_k_folds(training_data, k)

for i in range(k):
    tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
    tr_n, f_d = np.shape(tr_data[0])
    te_n, = np.shape(te_data[1])
    # TODO prepare distribution
    d = util.init_distribution(len(tr_data[0]))
    boost = b.Boosting(d)
    testing_predict = np.zeros((1, te_n)).tolist()[0]
    training_predict = np.zeros((1, tr_n)).tolist()[0]
    round_tr_err = []
    round_te_err = []
    converged = False
    tol = 1e-5
    test_err = 2.
    round = 0
    max_round = 1000
    while round < max_round:
        round += 1
        boost.add_model(ds.RandomDecisionStump, tr_data[0], tr_data[1], threshes)
        boost.update_predict(tr_data[0], training_predict)
        boost.update_predict(te_data[0], testing_predict)
        c_model_err = boost.model[-1].n_err
        print("Prediction 1: {}".format(testing_predict[0]))
        print("Model {} Round decision stump error: {}".format(round, c_model_err))
        c_tr_err = util.get_err_from_predict(training_predict, tr_data[1])
        print("Model {} Round training error: {}".format(round, c_tr_err))
        round_tr_err.append(c_tr_err)
        c_te_err = util.get_err_from_predict(testing_predict, te_data[1])
        print("Model {} Round testing error: {}".format(round, c_te_err))
        round_te_err.append(c_te_err)
        converged =  c_te_err / test_err > 1 - tol
        test_err = c_te_err

    training_errs.append(round_tr_err[-1])
    testing_errs.append(round_te_err[-1])


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
result['Fold'] = str(k)
result['Trainingerrs'] = str(training_errs)
result['MeanTrainingAcc'] = str(mean_training_err)
result['Testingerrs'] = str(testing_errs)
result['MeanTestingAcc'] = str(mean_testing_err)

result['ROC'] = str(roc)
result['AUC'] = str(auc)



# log the training result to file
util.write_result_to_file(result_path, model_name, result)