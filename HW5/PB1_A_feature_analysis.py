import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b
import profile

def main():
    # training parameter
    k = 10  # fold
    round_limit = 300
    result_path = 'results/PB1_A_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    threshes_path = 'data/spambase.threshes'
    data_path = 'data/spam/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    util.replace_zero_label_with_neg_one(training_data)

    # load thresholds
    threshes = loader.load_pickle_file(threshes_path)

    # start training
    training_errs = []
    testing_errs = []
    round_err_1st_boost = None
    tr_errs_1st_boost = None
    te_errs_1st_boost = None
    te_auc_1st_boost = None
    te_roc_1st_boost = None
    ranked_f = None
    roc = []
    auc = 0.0
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in range(1):
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
        tr_n, f_d = np.shape(tr_data[0])
        te_n, = np.shape(te_data[1])
        # TODO prepare distribution
        d = util.init_distribution(len(tr_data[0]))
        # TODO compute thresholds cheat sheet
        thresh_cs = util.pre_compute_threshes(tr_data[0], tr_data[1], threshes)
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
        while round < round_limit:  # and not converged:
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
            round_tr_err.append(c_tr_err)
            round_te_err.append(c_te_err)
            round_te_auc.append(c_te_auc)
            print('Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {:.12f} AUC {:.12f}'.format(round, c_f_ind, c_thresh, c_model_err, c_tr_err, c_te_err, c_te_auc))
            converged =  abs(c_te_auc - te_auc) / te_auc <= tol
            te_auc = c_te_auc

        training_errs.append(round_tr_err[-1])
        testing_errs.append(round_te_err[-1])
        if i == 0:
            ranked_f = util.get_f_ranking_from_predictions(boost, threshes)
            round_err_1st_boost = round_model_err
            tr_errs_1st_boost = round_tr_err
            te_errs_1st_boost = round_te_err
            te_auc_1st_boost = round_te_auc
            # TODO get feature ranking from the predictions

            _, te_roc_1st_boost = util.get_auc_from_predict(testing_predict, te_data[1], True)

        # break      # for testing

    mean_training_err = np.mean(training_errs)
    mean_testing_err = np.mean(testing_errs)

    print('Final results. Mean Train err: {}, Mean Test err: {}'.format(mean_training_err, mean_testing_err))
    print('Top 10 features: ')
    print(ranked_f[:10])

    result = {}
    result['Fold'] = k
    result['Trainingerrs'] = training_errs
    result['MeanTrainingAcc'] = mean_training_err
    result['Testingerrs'] = testing_errs
    result['MeanTestingAcc'] = mean_testing_err
    result['1stBoostTrainingError'] = tr_errs_1st_boost
    result['1stBoostTestingError'] = te_errs_1st_boost
    result['1stBoostModelError'] = round_err_1st_boost
    result['1stBoostTestingAUC'] = te_auc_1st_boost
    result['1stBoostTestingROC'] = te_roc_1st_boost
    result['rankedFeatures'] = ranked_f

    # result['ROC'] = str(roc)
    result['AUC'] = auc

    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()