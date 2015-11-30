import DataLoader as loader
import numpy as np
import Preprocess
import time
import MySVM as svm
import Utilities as util


def main():
    # training parameter
    k = 10  # fold
    round_limit = 300
    result_path = 'results/PB2_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    data_path = 'data/spam/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    util.replace_zero_label_with_neg_one(training_data)

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
    print('Preparing k fold data.')
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in range(1):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
        tr_n, f_d = np.shape(tr_data[0])
        te_n, = np.shape(te_data[1])

        # start training
        print('{:.2f} Start training.'.format(time.time() - st))
        c = 0.01
        tol = 0.01
        epsilon = 0.001
        # kernel = 'rbf'
        kernel = 'linear'
        clf = svm.SVM(C=c, tol=tol, epsilon=epsilon, kernel=kernel)
        clf.fit(tr_data[0], tr_data[1])
        tr_pred = clf.predict(tr_data[0])
        te_pred = clf.predict(te_data[0])

        tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
        te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

        print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))


if __name__ == '__main__':
    # profile.run('main()')
    main()