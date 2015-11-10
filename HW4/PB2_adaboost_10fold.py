import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b

def main():
    # training parameter
    target = 'crx'
    # target = 'vote'
    k = 10  # fold
    round_limit = 150

    if target == 'crx':
        result_path = 'results/crxBoosting_final_1.acc'
        model_name = 'crx_' + str(k) + 'fold'
        threshes_path = 'data/crx.threshes'
        data_path = 'data/crx_parsed.data'
    else:
        result_path = 'results/voteBoosting_final.acc'
        model_name = 'vote_' + str(k) + 'fold'
        threshes_path = 'data/vote.threshes'
        data_path = 'data/vote_parsed.data'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)

    # load thresholds
    threshes = loader.load_pickle_file(threshes_path)

    # start training
    training_errs = []
    testing_errs = []
    round_err_1st_boost = None
    tr_errs_1st_boost = None
    te_errs_1st_boost = None
    te_auc_1st_boost = None
    roc = []
    auc = 0.0
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in range(k):
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
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
            # c_te_auc = util.get_auc_from_predict(testing_predict, te_data[1])
            # round_tr_err.append(c_tr_err)
            # round_te_err.append(c_te_err)
            # round_te_auc.append(c_te_auc)
            print('Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {:.12f}'.format(round, c_f_ind, c_thresh, c_model_err, c_tr_err, c_te_err))
            # converged =  abs(c_te_auc - te_auc) / te_auc <= tol
            # te_auc = c_te_auc

        training_errs.append(c_tr_err)
        testing_errs.append(c_te_err)
        # if k == 0:
        #     round_err_1st_boost = round_model_err
        #     tr_errs_1st_boost = round_tr_err
        #     te_errs_1st_boost = round_te_err
            # te_auc_1st_boost = round_te_auc

        # break      # for testing


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
    result['1stBoostTrainingError'] = str(tr_errs_1st_boost)
    result['1stBoostTestingError'] = str(te_errs_1st_boost)
    result['1stBoostModelError'] = str(round_err_1st_boost)
    result['1stBoostTestingAUC'] = str(te_auc_1st_boost)

    # result['ROC'] = str(roc)
    result['AUC'] = str(auc)



    # log the training result to file
    util.write_result_to_file(result_path, model_name, result)

if __name__ == '__main__':
    # profile.run('main()')
    main()