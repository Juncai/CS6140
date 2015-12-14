import DataLoader as loader
import numpy as np
import Preprocess
import time
import MySVM as svm
import Utilities as util
import Consts as c


def main():
    # training parameter
    k = 10  # fold
    result_path = 'results/PB2_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    model_path = 'results/PB2_spam_mySVM_0a_norm_skip.pickle'
    data_path = 'data/spam/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    util.replace_zero_label_with_neg_one(training_data)

    # normalize feature
    Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, training_data[0])

    # start training
    print('Preparing k fold data.')
    k_folds = Preprocess.prepare_k_folds(training_data, k)
    tr_accs = []
    te_accs = []
    tt_st = time.time()

    for i in range(k):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)
        tr_n, f_d = np.shape(tr_data[0])
        te_n, = np.shape(te_data[1])

        # start training
        print('{:.2f} Start training.'.format(time.time() - st))
        cc = 0.01
        tol = 0.01
        epsilon = 0.001
        # kernel = c.GAUSSIAN
        kernel = c.LINEAR
        # kernel = c.RBF
        clf = svm.SVM(C=cc, tol=tol, epsilon=epsilon, kernel=kernel)
        clf.fit(tr_data[0], tr_data[1])
        tr_pred = clf.predict([tr_data[0][0]])
        te_pred = clf.predict(te_data[0])

        tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
        te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

        print('{} Results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))
        tr_accs.append(tr_acc)
        te_accs.append(te_acc)

        # save the svm
        # loader.save(model_path, clf)
    print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - tt_st, np.mean(tr_accs), np.mean(te_accs)))


if __name__ == '__main__':
    # profile.run('main()')
    main()