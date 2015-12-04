import DataLoader as loader
import numpy as np
import Preprocess
import time
import Utilities as util
import kNN
import Consts as c


def main():
    # training parameter
    k = 8  # fold
    result_path = 'results/PB2_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    data_path = 'data/spam/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    # util.replace_zero_label_with_neg_one(training_data)

    Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, training_data[0])
    # Preprocess.normalize_features_all(Preprocess.shift_and_scale, training_data[0])


    # start training
    training_accs = []
    testing_accs = []
    print('Preparing k fold data.')
    k_folds = Preprocess.prepare_k_folds(training_data, k)
    kernel = c.EUCLIDIAN
    sst = time.time()
    for i in (0,):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)

        # start training
        print('{:.2f} Start training.'.format(time.time() - st))
        for r in (2.5, 4.7,):
            clf = kNN.kNN(kernel=kernel)
            # clf.fit(training_data[0], training_data[1])
            clf.fit(tr_data[0], tr_data[1])
            # tr_pred = clf.predict(training_data[0], r=r)
            tr_pred = clf.predict(tr_data[0], r=r)
            te_pred = clf.predict(te_data[0], r=r)

            # tr_acc = (training_data[1] == tr_pred).sum() / training_data[0].shape[0]
            tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
            te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

            testing_accs.append(te_acc)
            print('{} {}-fold results with kernel {}, r={}. Train acc: {}, Test acc: {}'.format(time.time() - st, i, kernel, r, tr_acc, te_acc))


    # print('{} Final results with kernel {}. Test acc: {}'.format(time.time() - sst, kernel, np.array(testing_accs).mean()))



if __name__ == '__main__':
    # profile.run('main()')
    main()