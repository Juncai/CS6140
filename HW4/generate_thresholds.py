import DataLoader as loader
import pickle
import numpy as np

def generate_thresholds(features, thresh_path, meta_f=None):
    n, d = np.shape(features)
    threshes = []
    if meta_f is not None:
        for i in range(d):
            if meta_f[i][0]:
                # deal with discrete features
                threshes.append((True, list(meta_f[i][2].values())))
            else:
                # continuous features
                cur_f = [x[i] for x in features]
                cur_threshes = sorted_unique_values(cur_f)
                threshes.append((False, cur_threshes))
    else:
        for i in range(d):
            cur_f = [x[i] for x in features]
            cur_threshes = sorted_unique_values(cur_f)
            threshes.append(cur_threshes)

    with open(thresh_path, 'wb+') as f:
        pickle.dump(threshes, f)
    return threshes

def sorted_unique_values(feature):
    cur_max = np.max(feature)
    cur_min = np.min(feature)
    uniq_vals = np.unique(feature).tolist()
    cur_threshes = [cur_min - 0.1,] # add a threshold below all the values
    for j in range(len(uniq_vals) - 1):
        cur_threshes.append(np.mean(uniq_vals[j:j+2]))
    cur_threshes.append(cur_max + 0.1)  # add a threshold above all the values
    return cur_threshes

if __name__ == '__main__':
    # generate thresholds for spambase
    # data_path = 'data/spambase.data'
    # thresh_path = 'data/spambase.threshes'
    # data = loader.load_dataset(data_path)
    # generate_thresholds(data[0], thresh_path)

    # generate thresholds for crx
    # data_path = 'data/crx_parsed.data'
    # config_path = 'data/crx/crx.config'
    # thresh_path = 'data/crx.threshes'
    # data = loader.load_pickle_file(data_path)
    # config = loader.parse_UCI_config(config_path)
    # generate_thresholds(data[0], thresh_path, config[2])

    # generate thresholds for vote
    data_path = 'data/vote_parsed.data'
    config_path = 'data/vote/vote.config'
    thresh_path = 'data/vote.threshes'
    data = loader.load_pickle_file(data_path)
    config = loader.parse_UCI_config(config_path)
    generate_thresholds(data[0], thresh_path, config[2])