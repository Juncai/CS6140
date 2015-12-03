__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import MySVM as svm
import time

def main():
    # training parameter
    c = 0.01
    tol = 0.01
    epsilon = 0.001
    # kernel = 'rbf'
    kernel = 'linear'

    result_path = 'results/digits_ECOC_' + '_1.acc'
    model_name = 'digits_svm_2'
    model_path = 'results/' + model_name + '.model'
    te_pred_dict_path = 'results/digits_svm_test_pred_dict'
    tr_data_path = 'data\\digits\\tr_f_l.pickle'
    te_data_path = 'data\\digits\\te_f_l.pickle'
    # threshes_path = 'data\\digits\\sel_tr.threshes'


    # laod and preprocess training data
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data= loader.load_pickle_file(te_data_path)

    # transpose label
    # tr_data[1] = np.transpose(tr_data[1])[0]
    # te_data[1] = np.transpose(te_data[1])[0]

    # load thresholds
    # threshes = loader.load_pickle_file(threshes_path)

    # start training
    tr_n = len(tr_data[0])
    te_n = len(te_data[1])


    # train 45 svm
    print('Begin training...')
    svm_dict = {}  # list of svm classifiers


    function_tr_err = []

    # test the svms
    test_pred_dict = {}

    st = time.time()
    # prepare 45 datasets
    fn_count = 0
    for i in range(9):
        svm_dict[i] = {}
        test_pred_dict[i] = {}
        for j in range(i + 1, 10):
            if i == j:
                continue
            # get training data for this class
            c_tr_f, c_tr_y = data_i_j(tr_data[0], tr_data[1], i, j)
            # train svm
            print('{:.2f} Start training.'.format(time.time() - st))
            clf = svm.SVM(C=c, tol=tol, epsilon=epsilon, kernel=kernel)
            clf.fit(c_tr_f, c_tr_y)
            tr_pred = clf.predict(c_tr_f)

            tr_acc = (c_tr_y == tr_pred).sum() / c_tr_f.shape[0]

            fn_count += 1
            print('{} Function {} done. Final results. Train acc: {}'.format(time.time() - st, fn_count, tr_acc))

            svm_dict[i][j] = clf

            te_pred = clf.predict(te_data[0])
            test_pred_dict[i][j] = te_pred

    print('{} Training finished.'.format(time.time() - st))
    loader.save(model_path, svm_dict)
    loader.save(te_pred_dict_path, test_pred_dict)

def test():
    model_name = 'digits_svm_1'
    model_path = 'results/' + model_name + '.model'
    te_pred_dict_path = 'results/digits_svm_test_pred_dict'
    tr_data_path = 'data\\digits\\tr_f_l.pickle'
    te_data_path = 'data\\digits\\te_f_l.pickle'
    # threshes_path = 'data\\digits\\sel_tr.threshes'


    # laod and preprocess training data
    # tr_data = loader.load_pickle_file(tr_data_path)
    te_data= loader.load_pickle_file(te_data_path)
    model = loader.load_pickle_file(model_path)
    # te_pred_dict = loader.load_pickle_file(te_pred_dict_path)

    test_pred_dict = {}
    for i in range(9):
        test_pred_dict[i] = {}
        for j in range(i + 1, 10):
            if i == j:
                continue
            # get training data for this class
            clf = model[i][j]
            te_pred = clf.predict(te_data[0])
            test_pred_dict[i][j] = te_pred


    te_n = len(te_data[1])
    te_pred = np.zeros((1, te_n))[0]

    for i in range(te_n):
        votes = np.zeros((10,), dtype=np.int)
        for j in range(9):
            for k in range(j):
                votes[j] += 1 if test_pred_dict[k][j][i] == -1 else 0
            for kk in test_pred_dict[j]:
                votes[j] += 1 if test_pred_dict[j][kk][i] == 1 else 0
        count = np.bincount(votes)
        if count[-1] == 1:
            te_pred[i] = votes.argmax()
        else:
            te_pred[i] = votes.argmax()
            tie_ind = [votes.argmax()]
            cc = 0
            for ind_v, v in enumerate(votes):
                if v == votes.max():
                    if cc == 1:
                        tie_ind.append(ind_v)
                        break
                    else:
                        cc += 1
            te_pred[i] = tie_ind[0] if test_pred_dict[tie_ind[0]][tie_ind[1]][i] == 1 else tie_ind[1]
            print('{} Tie! {} wins.'.format(count[-1], te_pred[i]))


    acc = 0
    acc_n = 0
    for ind_l, l in enumerate(te_data[1]):
        acc += 1 if l == te_pred[ind_l] else 0

    acc /= te_n
    # acc = (te_data[1] == te_pred).sum() / te_n

    print('Acc: {}'.format(acc))



def data_i_j(tr_f, tr_l, i, j):
    '''
    i = 1, j = -1
    :param tr_f:
    :param tr_l:
    :param i:
    :param j:
    :return:
    '''
    res_f = []
    res_l = []
    for ind, l in enumerate(tr_l):
        if l == i:
            res_f.append(tr_f[ind])
            res_l.append(1)
        elif l == j:
            res_f.append(tr_f[ind])
            res_l.append(-1)
    return np.array(res_f), np.array(res_l)

# for ind, c_ecoc in enumerate(best_ecoc[1]):
#     print('Training function {}...'.format(ind))
#     # TODO preprocess labels, so that labels match ecoc, {0, 1} -> {-1, 1}
#     bin_label = util.generate_bin_label_from_ecoc(tr_data[1], c_ecoc)
#
#     # TODO prepare distribution
#     d = util.init_distribution(tr_n)
#     thresh_cs = None
#     if wl == ds.DecisionStump:
#         # TODO precompute thresholds cheat sheet
#         thresh_cs = util.pre_compute_threshes(tr_data[0], bin_label, threshes)
#     boost = b.Boosting(d)
#     training_predict = np.zeros((1, tr_n)).tolist()[0]
#     round_tr_err = []
#     round = 0
#     while round < max_round:
#         st = time.time() # start ts
#         round += 1
#         boost.add_model(wl, tr_data[0], bin_label, threshes, thresh_cs)
#         boost.update_predict(tr_data[0], training_predict)
#         c_model_err = boost.model[-1].w_err
#         c_tr_err = util.get_err_from_predict(training_predict, bin_label)
#         round_tr_err.append(c_tr_err)
#         c_f_ind = boost.model[-1].f_ind
#         c_thresh = boost.model[-1].thresh
#         print('Time used: {}'.format(time.time() - st))
#         print('Round: {} Feature: {} Threshold: {} Round_err: {:.12f} Train_err: {:.12f} Test_err {} AUC {}'.format(round, c_f_ind, c_thresh, c_model_err, c_tr_err, 0, 0))
#         if c_tr_err == 0:
#             break
#     function_tr_err.append(c_tr_err)
#     boosts.append(boost)
#
# print('Training done.')
#
# # TODO calculate ecoc prediction
# # training error
# train_err = util.ecoc_test(tr_data[0], tr_data[1], boosts, best_ecoc[2])
# test_err = util.ecoc_test(te_data[0], te_data[1], boosts, best_ecoc[2])
#
# print('Training err is: {}'.format(train_err))
# print('Testing err is: {}'.format(test_err))
# print('Training err for each function: ')
# print(str(function_tr_err))
#
# result = {}
# result['Testingerr'] = test_err
# result['Trainingerr'] = train_err
# result['RoundTrainingData'] = function_tr_err
# result['ECOC'] = best_ecoc[2]
#
# # save the model
# with open(model_path, 'wb+') as f:
#     pickle.dump(boosts, f)
#
# # log the training result to file
# util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # main()
    test()