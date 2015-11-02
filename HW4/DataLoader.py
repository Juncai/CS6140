import numpy as np
import pickle


def load_8news_data(path, save_path):
    features = []
    label = []
    l_ind = 0
    with open(path, 'r') as f:
        for line in f:
            raw = line.split()
            # the first column is label
            label.append(int(raw[0]))
            # parse the features
            features.append(parse_8news_features(raw[1:]))

    # save the data to the file
    with open(save_path, 'wb+') as f:
        pickle.dump([features, label], f)
    return features, label

def parse_8news_features(los):
    f_dict = {}
    for s in los:
        k_v = s.split(':')
        f_dict[int(k_v[0])] = float(k_v[1])
    return f_dict


def load_UCI_data(path, meta, save_path):
    '''
    Load UCI data according to the meta data, than save to the file
    :param path:
    :param meta:
    :return:
    '''
    n, n_f, meta_f, l_dict = meta
    features = []
    label = []
    with open(path, 'r') as f:
        for i in range(n):
            raw = f.readline().split()
            cur_f = []
            for j in range(n_f):
                if raw[j] != '?' and meta_f[j][0]:
                    cur_f.append(meta_f[j][2][raw[j]])
                else:
                    cur_f.append(raw[j])
            features.append(cur_f)
            label.append(l_dict[raw[-1]])

    # TODO deal with the missing value
    for i in range(n_f):
        if meta_f[i][0]:
            # for discrete feature
            value_count = {}
            for f in features:
                if f[i] in value_count:
                    value_count[f[i]] += 1
                else:
                    value_count[f[i]] = 1
            most_common = sorted(value_count, key = value_count.get, reverse = True)
            rep = most_common[0] if most_common[0] != '?' else most_common[1]
            for f in features:
                if f[i] == '?':
                    f[i] = rep
        else:
            # for continuous feature
            sum = 0.
            count = 0.
            for f in features:
                if f[i] != '?':
                    f[i] = float(f[i])
                    sum += f[i]
                    count += 1
            rep = sum / count
            for f in features:
                if f[i] == '?':
                    f[i] = rep

    with open(save_path, 'wb+') as f:
        pickle.dump([features, label], f)
    return [features, label, meta_f]


def parse_UCI_config(path):
    '''
    Parse the UCI data config file, return meta data of the data
    :param path:
    :return:
    '''
    meta_f = []
    with open(path, 'r') as f:
        n, n_d, n_c = [int(s) for s in f.readline().split()]
        n_f = n_d + n_c
        # read features' config
        for i in range(n_f):
            s_config = parse_single_config(f.readline())
            meta_f.append(s_config)
        # read labels' config
        l_dict = parse_label_config(f.readline())

    return n, n_f, meta_f, l_dict

def preprocess_features(values):
    v_dict = {}
    for ind, v in enumerate(values):
        v_dict[v] = ind
    return v_dict

def parse_label_config(config):
    s_list = config.split()
    l_dict = {}
    l_dict[s_list[1]] = -1
    l_dict[s_list[2]] = 1
    return l_dict

def parse_single_config(config):
    '''
    Parse single feature config
    :param config:
    :return:
    '''
    s_list = config.split()
    is_d = False
    value_num = None
    f_dict = None
    if len(s_list) > 1:
        is_d = True
        value_num = int(s_list[0])
        f_dict = {}
        p_values = s_list[1:]
        ind = 0
        for v in p_values:
            f_dict[v] = ind
            ind += 1
    return is_d, value_num, f_dict


def load_pickle_file(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_arrays(path):
    '''
    Read other data from file, no need to be matrix
    '''
    f = open(path, 'r')
    ar = []
    for line in f:
        ar.append([float(x) for x in line.split(',')])
    return ar


def load_dataset(path, has_label=True):
    '''
    Read data from a file and store them in a matrix
    '''
    f = open(path, 'r')
    deli = ','
    first_line = f.readline()
    if len(first_line.split(None)) > 1:
        # only works for more than one feature!
        deli = None
    f.close()
    data_set = np.genfromtxt(path, delimiter=deli)

    if has_label:
        label = [x[-1] for x in data_set]
        features = [x[:-1] for x in data_set]
        return [features, label]
    else:
        return (data_set, [])
