import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Model as m
import profile
import time

from sklearn.naive_bayes import GaussianNB


def main():
    st = time.time()
    # training parameter
    result_path = 'results/PB2_A_spam_polluted_NB_Gaussian.acc'
    model_name = 'spam_'
    train_data_path = 'data/spam_polluted/train/data.pickle'
    test_data_path = 'data/spam_polluted/test/data.pickle'

    # laod and preprocess training data
    tr_data = loader.load_pickle_file(train_data_path)
    te_data = loader.load_pickle_file(test_data_path)
    print('{:.2f} Data loaded!'.format(time.time() - st))

    # training with sklearn
    # gnb = GaussianNB()
    # tr_pred = gnb.fit(tr_data[0], tr_data[1]).predict(tr_data[0])
    # print('{} Training acc: {}'.format(time.time() - st, (tr_data[1] != tr_pred).sum() / tr_data[0].shape[0]))
    # te_pred = gnb.fit(tr_data[0], tr_data[1]).predict(te_data[0])
    # print('{} Testing acc: {}'.format(time.time() - st, (te_data[1] != te_pred).sum() / te_data[0].shape[0]))

    # start training
    model = m.NBGaussian()
    model.build(tr_data[0], tr_data[1])

    tr_pred = model.predict(tr_data[0])
    te_pred = model.predict(te_data[0])

    tr_acc = (tr_data[1] != tr_pred).sum() / tr_data[0].shape[0]
    te_acc = (te_data[1] != te_pred).sum() / te_data[0].shape[0]


    print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))

    result = {}
    result['TrainingAcc'] = tr_acc
    result['TestingAcc'] = te_acc

    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()