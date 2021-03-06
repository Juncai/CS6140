import numpy as np


def shift_and_scale(ds, col):
    '''
    normalize all features in the given dataset with shift and scale
    method
    '''
    tmp_ar = [x[col] for x in ds]
    min_val = np.amin(tmp_ar)
    max_val = np.amax(tmp_ar)
    range_val = max_val - min_val
    for i in range(len(tmp_ar)):
        ds[i][col] = (tmp_ar[i] - min_val) / range_val


def zero_mean_unit_var(ds, col):
    '''
    normalize all features in the given dataset with zero mean
    and unit variance method
    '''
    tmp_ar = [x[col] for x in ds]
    mean_val = np.mean(tmp_ar)
    std_val = np.std(tmp_ar)
    for i in range(len(tmp_ar)):
        ds[i][col] = (tmp_ar[i] - mean_val) / std_val


def normalize_features(methods, cols, train_ds, test_ds=None):
    '''
    normalize the given features with specified method for each col
    '''
    cur_ds = train_ds
    if test_ds:
        cur_ds = train_ds + test_ds

    for i, m in enumerate(methods):
        m(cur_ds, cols[i])
    if test_ds:
        train_ds = cur_ds[0:len(train_ds)]
        test_ds = cur_ds[len(train_ds):]


def normalize_features_all(method, train_ds, test_ds=None):
    '''
    Apply given normalize method to all of the feature columns
    '''
    if method is None:
        return
    cur_ds = train_ds
    if test_ds:
        cur_ds = train_ds + test_ds

    for i in range(len(cur_ds[0]) - 1):
        method(cur_ds, i)

    if test_ds:
        train_ds = cur_ds[0:len(train_ds)]
        test_ds = cur_ds[len(train_ds):]

# def preprocess_data(path, test_path=None):
#     (label, features) = file_to_dataset(path)
#     if test_path:
#         (label_test, features_test) = file_to_dataset(test_path)


def prepare_k_fold_data(dataset, k, ind):
    '''
    Prepare K-Fold training and testing data
    '''
    features = dataset[0]
    label = dataset[1]
    count = len(label)
    training_f = []
    training_l = []
    testing_f = []
    testing_l = []

    p1 = int(1.0 * (ind - 1) * count / k)
    p2 = int(1.0 * ind * count / k)
    training_f += features[0:p1] + features[p2:]
    training_l += label[0:p1] + label[p2:]
    testing_f += features[p1:p2]
    testing_l += label[p1:p2]
    return ((training_f, training_l), (testing_f, testing_l))
