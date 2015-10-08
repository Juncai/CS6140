import numpy as np
import math
import Consts as c

HOUSING_DELI = None
SPAM_DELI = ','

def acc_stable(theta, features, label, last_n_accs, thresh):
    x = [[1] + f for f in features]
    x = np.array(x)
    y = np.dot(x, theta)
    y = [yy[0] for yy in y]
    label = [l[0] for l in label]
    cur_acc = acc(y, label)
    print 'acc: ' + str(cur_acc)
    print 'mse: ' + str(mse(y, label))
    last_n_accs.append(cur_acc)
    if len(last_n_accs) > 10:
        last_n_accs.popleft()
    return np.var(last_n_accs) < thresh

def calculate_auc(roc):
    auc = 0
    for i in range(1, len(roc)):
        auc += (roc[i][0] - roc[i-1][0]) * (roc[i][1] + roc[i-1][1])
    return auc / 2.0

def confusion_matrix_mean(cms):
    sum = [0, 0, 0, 0]

    for cm in cms:
        for i in range(len(cm)):
            sum[i] += cm[i]
    mean = [s / len(cms) for s in sum]
    return mean

def compute_acc_confusion_matrix(predictions, labels, thresh=0.5):
    '''

    :param predictions:
    :param labels:
    :return: (acc, (TruePos, FalsePos, TrueNeg, FalseNeg))
    '''
    pos = 0
    neg = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            pos += 1
            if predictions[i] >= thresh:
                true_pos += 1
            else:
                false_neg += 1
        else:
            neg += 1
            if predictions[i] < thresh:
                true_neg += 1
            else:
                false_pos += 1
    acc = float(true_pos + true_neg) / len(labels)
    cm = (float(true_pos) / pos if pos != 0 else 1,
          float(false_pos) / neg if neg != 0 else 0,
          float(true_neg) / neg if neg != 0 else 1,
          float(false_neg) / pos if pos != 0 else 0)
    return acc, cm


def write_result_to_file(path, model_name, result):
    # Log the result
    f = open(path, 'w+')
    f.write('Model: ' + model_name + '\n')
    for k in result.keys():
        f.write(k + ' ' + str(result[k]) + '\n')
    f.close()
    return


def mse_for_nn(nn_outputs, exp_outputs):
    res = np.array(nn_outputs)
    exp = np.array(exp_outputs)
    diff = (res - exp).tolist()
    mse = 0
    for r in diff:
        for c in r:
           mse += math.pow(c, 2)
    # print mse
    return mse / len(nn_outputs)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def mistakes_less_than(m, thresh):
    return len(m) < thresh

def get_is_batch(config):
    if config[c.UPDATE_STYLE] == c.BATCH:
        return True
    else:
        return False

def get_term_fun(config):
    '''
    Get termination function for the config
    '''
    if config[c.TERM_CON] == c.MSE:
        return mse_less_than
    elif config[c.TERM_CON] == c.ACC:
        return acc_higher_than


def acc_higher_than(theta, features, label, thresh):
    '''
    Return True if the accuracy is higher than or equal to the threshold, False otherwise
    '''
    x = [[1] + f for f in features]
    x = np.array(x)
    y = np.dot(x, theta)
    y = [yy[0] for yy in y]
    label = [l[0] for l in label]
    cur_acc = acc(y, label)
    print 'acc: ' + str(cur_acc)
    print 'mse: ' + str(mse(y, label))
    return cur_acc >= thresh


def logistic_fun_batch(theta, features):
    '''
    Perform logistic regression calculation
    :param theta:
    :param features:
    :return:
    '''
    y = []
    for x in features:
        tmp = logistic_fun(theta, x)
        y.append([tmp])
    return y

def logistic_fun(theta, x):
    # x = x.tolist()
    # x = [1] + x
    wx = np.dot(x, theta)[0]
    return 1.0 / (1 + np.exp(-wx))



def mse_less_than(theta, features, label, thresh):
    '''
    Return True if the error is less than the threshold, False otherwise
    '''
    x = [[1] + f.tolist() for f in features]
    x = np.array(x)
    y = np.dot(x, theta)
    y = [yy[0] for yy in y]
    label = [l[0] for l in label]
    cur_mse = mse(y, label)
    print cur_mse
    return cur_mse < thresh

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
