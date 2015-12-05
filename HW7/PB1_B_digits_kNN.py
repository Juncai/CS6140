import DataLoader as loader
import numpy as np
import time
import kNN
import Consts as c
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity



def main():
    is_sklearn = True
    # kernel = c.COSINE
    kernel = c.GAUSSIAN
    # kernel = c.POLY
    # training parameter
    result_path = 'results/PB2_spam.acc'
    model_name = 'digits_' + kernel
    model_path = 'data/PB1_B_digits_sk_Gaussian.model'

    tr_data_path = 'data\\digits\\tr_f_l.pickle'
    te_data_path = 'data\\digits\\te_f_l.pickle'
    # laod and preprocess training data
    tr_data = loader.load_pickle_file(tr_data_path)
    te_data = loader.load_pickle_file(te_data_path)

    # transpose label
    tr_data[1] = np.transpose(tr_data[1])[0]
    te_data[1] = np.transpose(te_data[1])[0]


    # start training
    models = []
    st = time.time()

    # start training
    print('{:.2f} Start training.'.format(time.time() - st))

    for k in (1, 3, 7):
        if not is_sklearn:
            clf = kNN.kNN(kernel=kernel)
            clf.fit(tr_data[0], tr_data[1])
            # tr_pred = clf.predict(tr_data[0], k=k)
            te_pred = clf.predict(te_data[0], k=k)
        else:
            clf = KNeighborsClassifier(n_neighbors=k, metric=cosine_distances)
            clf.fit(tr_data[0], tr_data[1])
            # tr_pred = clf.predict(tr_data[0])
            te_pred = clf.predict(te_data[0])

        # tr_acc = (tr_data[1] == tr_pred).sum() / tr_data[0].shape[0]
        te_acc = (te_data[1] == te_pred).sum() / te_data[0].shape[0]
        models.append(clf)
        print('{} Final results with kernel {} and k={}. Train acc: {}, Test acc: {}'.format(time.time() - st, kernel, k, 0, te_acc))

    loader.save(model_path, models)



if __name__ == '__main__':
    # profile.run('main()')
    main()