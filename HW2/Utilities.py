import numpy as np
import math
import Consts as c

HOUSING_DELI = None
SPAM_DELI = ','


def get_is_batch(config):
    if config[c.UPDATE_STYLE] == c.BATCH:
        return True
    else:
        return False

def get_term_fun(config):
    '''
    TODO: add other methods
    '''
    if config[c.TERM_CON] == c.MSE:
        return mse_less_than

def mse_less_than(theta, features, label, thresh):
    '''
    Return True if the error less than the threshold, False otherwise
    '''
    x = [[1] + f for f in features]
    x = np.array(x)
    y = np.dot(x, theta)
    y = [yy[0] for yy in y]
    label = [l[0] for l in label]
    return mse(y, label) < thresh

def get_test_method(config):
    test_method = mse
    if c.TEST_METHOD in config.keys():
        if config[c.TEST_METHOD] == c.MSE_TEST:
            test_method = mse
        elif config[c.TEST_METHOD] == c.ACC_TEST:
            test_method = acc
    return test_method


def get_subtree_data(features, label, f_index, thresh):
    # generate features and label for subtrees
    f_new_1 = []
    l_new_1 = []
    f_new_2 = []
    l_new_2 = []
    for i in range(len(label)):
        if features[i][f_index] < thresh:
            f_new_1.append(features[i])
            l_new_1.append(label[i])
        else:
            f_new_2.append(features[i])
            l_new_2.append(label[i])
    return ((f_new_1, l_new_1), (f_new_2, l_new_2))


def find_majority(label):
    tmp_arr = [int(y) for y in label]
    tmp_arr = np.array(tmp_arr)
    counts = np.bincount(tmp_arr)
    return np.argmax(counts)


def split_on_mse(features, label, threshs, layer, term_con, term_thresh):
    '''
    Find the best pair based on IG
    '''
    # check the terminating condition
    if term_con == c.LAYER and layer > term_thresh:
        return (None, np.mean(label)), None, None
    if term_con == c.DATAPOINT and len(label) <= term_thresh:
        return (None, np.mean(label)), None, None

    best_pair = None
    cur_err = reg_error(label, np.mean(label))
    min_err = float('inf')
    for i in range(len(features[0])):
        for j in range(len(threshs[i])):
            f_cur = [x[i] for x in features]
            err_i = err_all(f_cur, label, threshs[i][j])
            if err_i < min_err:
                min_err = err_i
                best_pair = (i, threshs[i][j])

    if term_con == c.ERROR_DEC and cur_err - min_err < term_thresh:
        return (None, np.mean(label)), None, None

    left_data, right_data = get_subtree_data(features, label, best_pair[0], best_pair[1])

    # in case there is not actual split happened
    if len(left_data[0]) == 0 or len(right_data[0]) == 0:
        best_pair = (None, np.mean(label))
    return best_pair, left_data, right_data


def err_all(feature, label, thresh):
    res = 0
    y_x_1 = []
    y_x_2 = []
    f_len = len(feature)
    for i, x in enumerate(feature):
        if x < thresh:
            y_x_1.append(label[i])
        else:
            y_x_2.append(label[i])
    if len(y_x_1):
        y_x_1_mean = np.mean(y_x_1)
        # y_pre_1 = [y_x_1_mean for j in range(len(y_x_1))]
        res += reg_error(y_x_1, y_x_1_mean)
    if len(y_x_2):
        y_x_2_mean = np.mean(y_x_2)
        # y_pre_2 = [y_x_2_mean for j in range(len(y_x_2))]
        res += reg_error(y_x_2, y_x_2_mean)
    return res


def reg_error(labels, predict):
    if len(labels) == 0:
        return 0
    res = 0.0
    for l in labels:
        res += math.pow(predict - l, 2)
    return res


def acc(predictions, labels, thresh=0.5):
    '''
    Compute the accuracy for binary label
    '''
    count = 0
    for i, p in enumerate(predictions):
        if p < thresh:
            if labels[i] == 0:
                count += 1
        elif labels[i] == 1:
            count += 1
    return 1.0 * count / len(labels)



def mse(predictions, labels):
    '''
    Calculate the error between predictions and actual labels
    '''
    if len(labels) == 0:
        return 0
    res = 0.0
    for i in range(len(labels)):
        res += math.pow(predictions[i] - labels[i], 2)
    return res / len(labels)


def split_on_ig(features, label, threshs, layer, term_con, term_thresh):
    '''
    Find the best pair based on IG
    '''
    best_pair = None
    if layer > term_thresh or compute_entropy(label) == 0:
        return (None, find_majority(label)), None, None
    max_ig = 0
    for i in range(len(features[0])):
        for j in range(len(threshs[i])):
            f_cur = [x[i] for x in features]
            ig_i = compute_ig(f_cur, label, threshs[i][j])
            if ig_i > max_ig:
                max_ig = ig_i
                best_pair = (i, threshs[i][j])

    if best_pair is None:
        return (None, find_majority(label)), None, None

    left_data, right_data = get_subtree_data(features, label, best_pair[0], best_pair[1])

    # in case there is not actual split happened
    if len(left_data[0]) == 0 or len(right_data[0]) == 0:
        best_pair = (None, find_majority(label))
    return best_pair, left_data, right_data


def compute_ig(feature, label, thresh):
    '''
    Compute Information Gain (Y|X)
    '''
    # compute H(Y)
    h_y = compute_entropy(label)

    # compute H(Y|X)
    f_len = len(feature)
    h_y_x = 0
    # x_1 = []
    y_x_1 = []
    # x_2 = []
    y_x_2 = []
    for i, x in enumerate(feature):
        if x < thresh:
            # x_1.append(feature[i])
            y_x_1.append(label[i])
        else:
            # x_2.append(feature[i])
            y_x_2.append(label[i])
    h_y_x += 1.0 * len(y_x_1) / f_len * compute_entropy(y_x_1)
    h_y_x += 1.0 * len(y_x_2) / f_len * compute_entropy(y_x_2)
    return h_y - h_y_x


def compute_entropy(label):
    '''
    Compute entropy of the label
    '''
    h = 0
    uniq_labels = np.unique(label)
    label_counts = np.bincount(label)
    for y in uniq_labels:
        p_y = 1.0 * label_counts[y] / len(label)
        h += p_y * np.log2(1.0 / p_y)
    return h


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', required=True)
    # opts = parser.parse_args()
    # file_to_dataset(opts.f)
