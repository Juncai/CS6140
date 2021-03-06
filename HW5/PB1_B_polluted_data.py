import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b
import profile
import time


def main():
    st = time.time()
    # training parameter
    round_limit = 15
    result_path = 'results/PB1_B_spam_2.acc'
    model_name = 'spam_'
    model_path = result_path + '.model'
    threshes_path = 'data/spambase_polluted.threshes'
    train_data_path = 'data/spam_polluted/train/data.pickle'
    test_data_path = 'data/spam_polluted/test/data.pickle'

    # laod and preprocess training data
    tr_data = loader.load_pickle_file(train_data_path)
    te_data = loader.load_pickle_file(test_data_path)
    print('{:.2f} Data loaded!'.format(time.time() - st))
    # TODO convert labels from {0, 1} to {-1, 1}
    util.replace_zero_label_with_neg_one(tr_data)
    util.replace_zero_label_with_neg_one(te_data)
    print('{:.2f} Label converted!'.format(time.time() - st))

    # load thresholds
    threshes = loader.load_pickle_file(threshes_path)
    print('{:.2f} Thresholds loaded!'.format(time.time() - st))
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
    thresh_cs = None

    tr_n, f_d = np.shape(tr_data[0])
    te_n, = np.shape(te_data[1])
    # TODO prepare distribution
    d = util.init_distribution(len(tr_data[0]))

    # TODO compute thresholds cheat sheet (not a solution due to huge thresh_cs table)
    # thresh_cs = util.pre_compute_threshes(tr_data[0], tr_data[1], threshes)
    # print('{:.2f} Thresholds cheat sheet computed!'.format(time.time() - st))

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
        # c_te_auc = util.get_auc_from_predict(testing_predict, te_data[1])
        round_tr_err.append(c_tr_err)
        round_te_err.append(c_te_err)
        # round_te_auc.append(c_te_auc)
        print('{:.2f} Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {:.12f} AUC {:.12f}'.format(time.time() - st, round, c_f_ind, c_thresh, c_model_err, c_tr_err, c_te_err, 0))
        # converged =  abs(c_te_auc - te_auc) / te_auc <= tol
        # te_auc = c_te_auc

    training_errs.append(round_tr_err[-1])
    testing_errs.append(round_te_err[-1])
    # TODO get feature ranking from the predictions
    ranked_f = util.get_f_ranking_from_predictions(boost, threshes)
    round_err_1st_boost = round_model_err
    tr_errs_1st_boost = round_tr_err
    te_errs_1st_boost = round_te_err
    # te_auc_1st_boost = round_te_auc

    # _, te_roc_1st_boost = util.get_auc_from_predict(testing_predict, te_data[1], True)

        # break      # for testing

    mean_training_err = np.mean(training_errs)
    mean_testing_err = np.mean(testing_errs)

    print('Final results. Mean Train err: {}, Mean Test err: {}'.format(mean_training_err, mean_testing_err))
    print('Top 10 features: ')
    # print(ranked_f[:10])

    result = {}
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

    # store the model
    loader.save(model_path, boost)
    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()