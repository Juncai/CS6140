import DataLoader as loader
import pickle
import numpy as np

def generate_thresholds_8news_v2(features, thresh_path):
    '''

    :param features:
    :param thresh_path:
    :return:
    '''
    # TODO find all the feature numbers appear in the data
    f_set = set()
    for f in features:
        for k in f.keys():
            f_set.add(k)
    f_list = list(f_set)
    f_list.sort()

    # TODO for each feature, establish the thresholds list
    # TODO and the features which contains the thresholds
    f_threshs = {}
    for f_k in f_list:
        # cur_f = [f[f_k] for f in features if f_k in f.keys()]
        # cur_f_i = [i for i, f in features if f_k in f.keys()]
        cur_f = []
        cur_f_i = []
        cur_t = []
        for f_i, f in enumerate(features):
            if f_k in f.keys():
                cur_f.append(f[f_k])
                cur_f_i.append(f_i)
        cur_f = np.unique(cur_f).tolist()
        cur_f = [0] + cur_f     # prepend a zero to the feature values
        cur_t.append(-1)    # a threshold below all the values
        for i in range(len(cur_f) - 1):
            cur_t.append(np.mean(cur_f[i:i+2]))
        cur_t.append(cur_f[-1] + 0.1)   # a threshold above all the values
        f_threshs[f_k] = (cur_t, cur_f_i)

    with open(thresh_path, 'wb+') as f:
        pickle.dump(f_threshs, f)
    return f_threshs

def generate_thresholds_8news(features, thresh_path):
    '''

    :param features:
    :param thresh_path:
    :return:
    '''
    # TODO find all the feature numbers appear in the data
    f_set = set()
    for f in features:
        for k in f.keys():
            f_set.add(k)
    f_list = list(f_set)
    f_list.sort()

    # TODO for each feature, establish the thresholds list
    f_threshs = {}
    for f_k in f_list:
        cur_f = []
        cur_t = []
        for f in features:
            if f_k in f.keys():
                cur_f.append(f[f_k])
        cur_f = np.unique(cur_f).tolist()
        cur_f = [0] + cur_f     # prepend a zero to the feature values
        cur_t.append(-1)    # a threshold below all the values
        for i in range(len(cur_f) - 1):
            cur_t.append(np.mean(cur_f[i:i+2]))
        cur_t.append(cur_f[-1] + 0.1)   # a threshold above all the values
        f_threshs[f_k] = cur_t

    with open(thresh_path, 'wb+') as f:
        pickle.dump(f_threshs, f)
    return f_threshs

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
    # data_save_path = 'data/spam/data.pickle'
    # data_path = 'data/spam/spambase.data'
    # thresh_path = 'data/spambase.threshes'
    # data = loader.load_dataset(data_path)
    # generate_thresholds(data[0], thresh_path)
    # loader.save(data_save_path, data)

    # generate thresholds for spambase polluted train data
    # data_path = 'data/spam_polluted/train/data.pickle'
    # features_path = 'data/spam_polluted/train/train_feature.txt'
    # thresh_path = 'data/spambase_polluted.threshes'
    # label_path = 'data/spam_polluted/train/train_label.txt'
    # features = loader.load_dataset(features_path, False)
    # # generate_thresholds(features, thresh_path)
    # label = loader.load_label(label_path)
    # loader.save(data_path, [features, label])

    # load and save spam polluted test data
    # data_save_path = 'data/spam_polluted/test/data.pickle'
    # features_path = 'data/spam_polluted/test/test_feature.txt'
    # label_path = 'data/spam_polluted/test/test_label.txt'
    # features = loader.load_dataset(features_path, False)
    # label = loader.load_label(label_path)
    # loader.save(data_save_path, [features, label])

    # load and save spam polluted missing test data
    # data_save_path = 'data/spam_polluted_missing/test/data.pickle'
    # data_path = 'data/spam_polluted_missing/test/20_percent_missing_test.txt'
    # data = loader.load_dataset(data_path)
    # loader.save(data_save_path, data)

    # load and save spam polluted missing train data
    data_save_path = 'data/spam_polluted_missing/train/data.pickle'
    data_path = 'data/spam_polluted_missing/train/20_percent_missing_train.txt'
    data = loader.load_dataset(data_path)
    loader.save(data_save_path, data)



    # generate thresholds for crx
    # data_path = 'data/crx_parsed.data'
    # config_path = 'data/crx/crx.config'
    # thresh_path = 'data/crx.threshes'
    # data = loader.load_pickle_file(data_path)
    # config = loader.parse_UCI_config(config_path)
    # generate_thresholds(data[0], thresh_path, config[2])

    # generate thresholds for vote
    # data_path = 'data/vote_parsed.data'
    # config_path = 'data/vote/vote.config'
    # thresh_path = 'data/vote.threshes'
    # data = loader.load_pickle_file(data_path)
    # config = loader.parse_UCI_config(config_path)
    # generate_thresholds(data[0], thresh_path, config[2])

