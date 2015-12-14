import DataLoader as loader
import Preprocess
import time
import kNN
import Consts as c
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def main():
    # training parameter
    is_sklearn = True
    k = 10  # fold
    result_path = 'results/PB2_spam.acc'
    model_name = 'spam_' + str(k) + 'fold'
    data_path = 'data/spam/data.pickle'

    # laod and preprocess training data
    training_data = loader.load_pickle_file(data_path)
    # TODO convert labels from {0, 1} to {-1, 1}
    # util.replace_zero_label_with_neg_one(training_data)

    # Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, training_data[0])
    # training_data[0] = preprocessing.scale(training_data[0])


    # start training
    training_errs = []
    testing_errs = []
    print('Preparing k fold data.')
    k_folds = Preprocess.prepare_k_folds(training_data, k)

    for i in (0,):
        st = time.time()
        tr_data, te_data = Preprocess.get_i_fold(k_folds, i)

        # start training
        print('{:.2f} Start training.'.format(time.time() - st))
        kernel = c.EUCLIDEAN
        # kernel = c.GAUSSIAN
        f_select = True
        best_features_num = 5
        clf = kNN.kNN(kernel=kernel)
        clf.fit(tr_data[0], tr_data[1], f_select=f_select, best_f=best_features_num)
        print("Best features: {}".format(clf.best_f_indices))
        for kk in (1, 2, 3, 7):
            tr_pred = clf.predict(tr_data[0], k=kk)
            te_pred = clf.predict(te_data[0], k=kk)

            tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
            te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]

            print('{} Final results with kernel {}, k={}. Train acc: {}, Test acc: {}'.format(time.time() - st, kernel, kk, tr_acc, te_acc))





if __name__ == '__main__':
    # profile.run('main()')
    main()