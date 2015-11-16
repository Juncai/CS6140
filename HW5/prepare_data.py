import DataLoader as loader
import numpy as np
import math

def compute_feature_mean(features, save_path):
    n, d = np.shape(features)
    means = []
    for i in range(d):
        cur_mean = 0
        for f in features:
            if not math.isnan(f[i]):
                cur_mean += f[i]
        means.append(cur_mean / n)
    loader.save(save_path, means)
    return means


if __name__ == '__main__':
    # generate means for the features
    path = 'data/spam_polluted_missing/train/data.pickle'
    mean_path = 'data/spam_polluted_missing/train/f_mean.pickle'
    features = loader.load_pickle_file(path)[0]
    compute_feature_mean(features, mean_path)
