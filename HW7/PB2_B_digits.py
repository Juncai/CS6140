import DataLoader as loader
import numpy as np
import Preprocess
import time
import Utilities as util
import kde
import Consts as c


def main():
    # kernel = c.GAUSSIAN
    kernel = c.POLY
    # training parameter
    result_path = 'results/PB2_spam.acc'
    model_name = 'digits_' + kernel

    tr_data_path = 'data\\digits\\tr_f_l_10.pickle'
    te_data_path = 'data\\digits\\te_f_l_10.pickle'
    # laod and preprocess training data
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data = loader.load_pickle_file(te_data_path)

    # transpose label
    tr_data[1] = np.transpose(tr_data[1])[0]
    te_data[1] = np.transpose(te_data[1])[0]

    Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, tr_data[0])
    Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, te_data[0])

    # start training

    st = time.time()

    # start training
    print('{:.2f} Start training.'.format(time.time() - st))

    clf = kde.KDE(kernel=kernel)
    clf.fit(tr_data[0], tr_data[1])
    tr_pred = clf.predict(tr_data[0])
    te_pred = clf.predict(te_data[0])

    tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
    te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

    print('{} Final results with kernel {} Train acc: {}, Test acc: {}'.format(time.time() - st, kernel, tr_acc, te_acc))





if __name__ == '__main__':
    # profile.run('main()')
    main()