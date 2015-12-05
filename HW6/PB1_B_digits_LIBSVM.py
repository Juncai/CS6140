import DataLoader as loader
import numpy as np
import Preprocess
from sklearn import svm
import time


def main():
    # training parameter
    result_path = 'results/PB1_B_digits.acc'
    model_name = 'digits_'
    threshes_path = 'data/spambase.threshes'
    tr_data_path = 'data\\digits\\tr_f_l_10r.pickle'
    te_data_path = 'data\\digits\\te_f_l_10r.pickle'
    # laod and preprocess training data
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data = loader.load_pickle_file(te_data_path)

    # transpose label
    tr_data[1] = np.transpose(tr_data[1])[0]
    te_data[1] = np.transpose(te_data[1])[0]

    # start training
    # kernel = 'poly'
    kernel = 'linear'
    tol = 0.01
    c = 0.01

    st = time.time()

    # start training
    print('{} Start training. Kernel: {}'.format(time.time() - st, kernel))
    # clf = svm.SVC(kernel='poly')
    clf = svm.SVC(C=c, kernel=kernel, tol=tol)
    # clf = svm.NuSVC(kernel=kernel)
    clf.fit(tr_data[0], tr_data[1])
    tr_pred = clf.predict(tr_data[0])
    te_pred = clf.predict(te_data[0])

    tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
    te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

    print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))


    #     boost = b.Boosting(d)
    #     testing_predict = np.zeros((1, te_n)).tolist()[0]
    #     training_predict = np.zeros((1, tr_n)).tolist()[0]
    #     round_tr_err = []
    #     round_te_err = []
    #     round_model_err = []
    #     round_te_auc = []
    #     converged = False
    #     tol = 1e-5
    #     te_auc = 2.
    #     round = 0
    #     while round < round_limit:  # and not converged:
    #         round += 1
    #         boost.add_model(ds.DecisionStump, tr_data[0], tr_data[1], threshes, thresh_cs)
    #         boost.update_predict(tr_data[0], training_predict)
    #         boost.update_predict(te_data[0], testing_predict)
    #         c_model_err = boost.model[-1].w_err
    #         round_model_err.append(c_model_err)
    #         c_f_ind = boost.model[-1].f_ind
    #         c_thresh = boost.model[-1].thresh
    #         c_tr_err = util.get_err_from_predict(training_predict, tr_data[1])
    #         c_te_err = util.get_err_from_predict(testing_predict, te_data[1])
    #         # TODO calculate the AUC for testing results
    #         c_te_auc = util.get_auc_from_predict(testing_predict, te_data[1])
    #         round_tr_err.append(c_tr_err)
    #         round_te_err.append(c_te_err)
    #         round_te_auc.append(c_te_auc)
    #         print('Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {:.12f} AUC {:.12f}'.format(round, c_f_ind, c_thresh, c_model_err, c_tr_err, c_te_err, c_te_auc))
    #         converged =  abs(c_te_auc - te_auc) / te_auc <= tol
    #         te_auc = c_te_auc
    #
    #     training_errs.append(round_tr_err[-1])
    #     testing_errs.append(round_te_err[-1])
    #     if i == 0:
    #         ranked_f = util.get_f_ranking_from_predictions(boost, threshes)
    #         round_err_1st_boost = round_model_err
    #         tr_errs_1st_boost = round_tr_err
    #         te_errs_1st_boost = round_te_err
    #         te_auc_1st_boost = round_te_auc
    #         # TODO get feature ranking from the predictions
    #
    #         _, te_roc_1st_boost = util.get_auc_from_predict(testing_predict, te_data[1], True)
    #
    #     # break      # for testing
    #
    # mean_training_err = np.mean(training_errs)
    # mean_testing_err = np.mean(testing_errs)
    #
    # print('Final results. Mean Train err: {}, Mean Test err: {}'.format(mean_training_err, mean_testing_err))
    # print('Top 10 features: ')
    # print(ranked_f[:10])
    #
    # result = {}
    # result['Fold'] = k
    # result['Trainingerrs'] = training_errs
    # result['MeanTrainingAcc'] = mean_training_err
    # result['Testingerrs'] = testing_errs
    # result['MeanTestingAcc'] = mean_testing_err
    # result['1stBoostTrainingError'] = tr_errs_1st_boost
    # result['1stBoostTestingError'] = te_errs_1st_boost
    # result['1stBoostModelError'] = round_err_1st_boost
    # result['1stBoostTestingAUC'] = te_auc_1st_boost
    # result['1stBoostTestingROC'] = te_roc_1st_boost
    # result['rankedFeatures'] = ranked_f
    #
    # # result['ROC'] = str(roc)
    # result['AUC'] = auc
    #
    # # log the training result to file
    # util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()