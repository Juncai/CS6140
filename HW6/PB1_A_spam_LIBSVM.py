import DataLoader as loader
import numpy as np
import Preprocess
from sklearn import svm
import time


def main():
    # training parameter
    k = 10  # fold
    round_limit = 300
    result_path = 'results/PB1_A_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    threshes_path = 'data/spambase.threshes'
    data_path = 'data/spam/data.pickle'
    kernel = 'poly'
    # kernel = 'linear'
    # kernel = 'rbf'
    verbose = False

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    # util.replace_zero_label_with_neg_one(training_data)


    print('Preparing k fold data.')
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in range(k):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)

        # start training
        print('{:3f} Start training. Kernel: {}'.format(time.time() - st, kernel))
        # clf = svm.SVC(kernel=kernel)
        clf = svm.NuSVC(kernel=kernel, tol=0.01, verbose=verbose)
        clf.fit(tr_data[0], tr_data[1])
        tr_pred = clf.predict(tr_data[0])
        te_pred = clf.predict(te_data[0])

        tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
        te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

        print('{:3f} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))


    # # log the training result to file
    # util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    main()