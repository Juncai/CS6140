import DataLoader as loader
import numpy as np
import Utilities as util
import Model as m
import time
from sklearn.decomposition import PCA


def main():
    st = time.time()
    # training parameter
    result_path = 'results/PB2_B_spam_polluted_final.acc'
    model_name = 'spam_'
    train_data_path = 'data/spam_polluted/train/data.pickle'
    test_data_path = 'data/spam_polluted/test/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(train_data_path)
    testing_data = loader.load_pickle_file(test_data_path)
    print('{:.2f} Data loaded!'.format(time.time() - st))

    # PCA step
    tr_features = np.array(training_data[0])
    te_features = np.array(testing_data[0])
    pca = PCA(n_components=100)
    tr_features = pca.fit_transform(tr_features)
    te_features = pca.transform(te_features)
    tr_data = [tr_features, training_data[1]]
    te_data = [te_features, testing_data[1]]

    # start training
    model = m.NBGaussian()
    model.build(tr_data[0], tr_data[1])

    print('{:.2f} Predicting...'.format(time.time() - st))
    tr_pred = model.predict(tr_data[0])
    te_pred = model.predict(te_data[0])

    print('{:.2f} Calculating results...'.format(time.time() - st))
    tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
    te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

    print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, tr_acc, te_acc))

    result = {}
    result['TrainingAcc'] = tr_acc
    result['TestingAcc'] = te_acc

    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()