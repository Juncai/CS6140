import DataLoader as loader
import numpy as np
import Preprocess
import time
import Utilities as util
import kNN
import Consts as c


def main():
    # kernel = c.COSINE
    kernel = c.GAUSSIAN
    # kernel = c.POLY
    # training parameter
    result_path = 'results/PB2_spam.acc'
    model_name = 'digits_' + kernel

    tr_data_path = 'data\\digits\\tr_f_l.pickle'
    te_data_path = 'data\\digits\\te_f_l.pickle'
    # laod and preprocess training data
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data = loader.load_pickle_file(te_data_path)

    # transpose label
    tr_data[1] = np.transpose(tr_data[1])[0]
    te_data[1] = np.transpose(te_data[1])[0]


    # start training

    st = time.time()

    # start training
    print('{:.2f} Start training.'.format(time.time() - st))

    for k in (1, 3, 7):
        clf = kNN.kNN(kernel=kernel)
        clf.fit(tr_data[0], tr_data[1])
        tr_pred = clf.predict(tr_data[0], k=k)
        te_pred = clf.predict(te_data[0], k=k)

        tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
        te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

        print('{} Final results with kernel {} and k={}. Train acc: {}, Test acc: {}'.format(time.time() - st, kernel, k, tr_acc, te_acc))





if __name__ == '__main__':
    # profile.run('main()')
    main()