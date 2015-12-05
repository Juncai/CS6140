__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import MySVM as svm
import time
import Utilities as util
from scipy.spatial.distance import hamming

result_path = 'results/digits_ECOC_' + '_1.acc'
model_name = 'digits_svm_ecoc_1'
model_path = 'results/' + model_name + '.model'
te_pred_dict_path = 'results/digits_svm_test_pred_dict_10r_c01'
# tr_data_path = 'data\\digits\\tr_f_l.pickle'
tr_data_path = 'data\\digits\\tr_f_l_10.pickle'
# te_data_path = 'data\\digits\\te_f_l.pickle'
te_data_path = 'data\\digits\\te_f_l_10.pickle'
# threshes_path = 'data\\digits\\sel_tr.threshes'
ecoc_path = 'data\\digits\\ecoc_cs'


def main():
    # training parameter
    c = 0.1
    tol = 0.01
    epsilon = 0.001
    # kernel = 'rbf'
    kernel = 'linear'

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


def ecoc():

    # training parameter
    c = 0.001
    tol = 0.01
    epsilon = 0.001
    # kernel = 'rbf'
    kernel = 'linear'

    # laod and preprocess training data
    print('Loading data...')
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data= loader.load_pickle_file(te_data_path)

    # randomly generate ECOC of 50 functions
    num_ecoc = 10
    class_num = 10
    best_ecoc = util.get_ecoc(ecoc_path, num_ecoc, class_num)

    # train 10 svm
    print('Begin training...')
    svms = []  # list of svm classifiers
    function_tr_err = []
    sst = time.time()
    for ind, c_ecoc in enumerate(best_ecoc[1]):
        st = time.time()
        # prepare label
        c_label = [-1 if c_ecoc[l] == 0 else 1 for l in tr_data[1]]
        clf = svm.SVM(C=c, tol=tol, epsilon=epsilon, kernel=kernel)
        clf.fit(tr_data[0], c_label)
        tr_pred = clf.predict(tr_data)
        tr_acc = (c_label == tr_pred).sum() / tr_data[0].shape[0]
        print('{} Function {} done. Final results. Train acc: {}'.format(time.time() - st, ind, tr_acc))
        svms.append(clf)

    print('{} Training finished.'.format(time.time() - sst))
    loader.save(model_path, svms)

def ecoc_test():
    svms = loader.load_pickle_file(model_path)
    te_data= loader.load_pickle_file(te_data_path)
    pred = []

    for f in te_data[0]:
        min_hamming_dist = 1.
        match_label = 0
        code = []
        for s in svms:
            c_pred = s.predict([f])[0]
            code.append(1 if c_pred == 1 else 0)  # replace -1 with 0
        for ind, c in enumerate(ecoc):
            cur_hd = hamming(c, code)
            if cur_hd < min_hamming_dist:
                min_hamming_dist = cur_hd
                match_label = ind
        pred.append(match_label)

    return (pred == te_data[1]).sum() / len(te_data[1])

def test():

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

if __name__ == '__main__':
    # main()
    # test()
    ecoc()