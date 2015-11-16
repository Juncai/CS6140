import DataLoader as loader
import numpy as np
import math

def compute_feature_mean(features, save_path):
    n, d = np.shape(features)

    means = []
    for i in range(d):

        cur_f = features[:, i]
        means.append(np.nanmean(cur_f))

        # cur_mean = 0
        # for f in features:
        #     if not np.isnan(f[i]):
        #         cur_mean += f[i]
        # means.append(cur_mean / n)
    means = np.array(means)
    loader.save(save_path, means)
    return means


if __name__ == '__main__':
    # generate means for the features, missing
    path = 'data/spam_polluted_missing/train/data.pickle'
    mean_path = 'data/spam_polluted_missing/train/f_mean.pickle'
    features = loader.load_pickle_file(path)[0]
    means = np.nanmean(features, axis=0)
    loader.save(mean_path, means)

    # generate means for the features, polluted
    # path = 'data/spam_polluted/train/data.pickle'
    # mean_path = 'data/spam_polluted/train/f_mean.pickle'
    # features = loader.load_pickle_file(path)[0]
    # means = np.nanmean(features, axis=0)
    # loader.save(mean_path, means)
