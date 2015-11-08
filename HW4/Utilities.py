import numpy as np
import math
import numpy.random as random
from scipy.spatial.distance import hamming
import copy

def generate_bin_label_from_ecoc(label, ecoc):
    bin_label = copy.deepcopy(label)
    pos_ind = []

    for ind, v in enumerate(ecoc):
        if v == 1:
            pos_ind.append(ind)

    for j, l in enumerate(bin_label):
        bin_label[j] = 1 if l in pos_ind else -1

    return bin_label

def ecoc_test(features, label, boosts, ecoc):
    pred = ecoc_prediction(features, boosts, ecoc)
    return acc_exact(pred, label)

def ecoc_prediction(features, boosts, ecoc):
    pred = []
    for f in features:
        c_pred = ecoc_prediction_single(f, boosts, ecoc)
        # print(c_pred)
        pred.append(c_pred)
    return pred


def ecoc_prediction_single(feature, boosts, ecoc):
    '''

    :param feature:
    :param boosts:
    :param ecoc: ecoc for predicting
    :return:
    '''
    min_hamming_dist = 1.
    match_label = 0
    code = []
    for b in boosts:
        c_pred = b.predict_single(feature)

        code.append(1 if c_pred == 1 else 0)  # replace -1 with 0
    for ind, c in enumerate(ecoc):
        cur_hd = hamming(c, code)
        if cur_hd < min_hamming_dist:
            min_hamming_dist = cur_hd
            match_label = ind
    return match_label


def get_bagging_data(ds, n):
    '''

    :param ds:
    :param n:
    :return:
    '''
    n_array = [i for i in range(len(ds[0]))]
    selected_inds = random.choice(n_array, n, replace=True)
    res = [[], []]
    for ind in selected_inds:
        res[0].append(ds[0][ind])
        res[1].append(ds[1][ind])
    return res

def pre_compute_threshes_8news(features, label, threshes):
    '''

    :param features:
    :param label:
    :param threshes:
    :return:
    '''
    thresh_cs = {}
    label_plus_one = np.array(label) + 1
    for t_k in threshes.keys():
        c_cs = []
        cur_f = np.array([(f[t_k] if t_k in f.keys() else 0) for f in features])
        for t_ind, t in enumerate(threshes[t_k]):
            cur_r = np.sign(cur_f - t) + 1
            cur_r = np.logical_xor(cur_r, label_plus_one)
            c_cs.append(cur_r)
        thresh_cs[t_k] = c_cs
    return thresh_cs


def pre_compute_threshes(features, label, threshes):
    '''

    :param features:
    :param label:
    :param threshes:
    :return:
    '''
    threshes_cheatsheet = []
    n, dim = np.shape(features)
    label_plus_one = np.array(label) + 1
    for i in range(dim):
        cur_f = np.array([x[i] for x in features])

        # sorted(cur_f, key= lambda x: x[0])
        c_cs = []
        for t in threshes[i]:
            cur_r = np.sign(cur_f - t) + 1
            cur_r = np.logical_xor(cur_r, label_plus_one)
            # cur_r = [1 if rr else 0 for rr in cur_r]
            c_cs.append(cur_r)
        threshes_cheatsheet.append(c_cs)
    return threshes_cheatsheet

def pre_compute_threshes_uci(features, label, threshes):
    '''

    :param features:
    :param label:
    :param threshes:
    :return:
    '''
    threshes_cheatsheet = []
    n, dim = np.shape(features)
    label_plus_one = np.array(label) + 1
    n_ones = np.ones((1, n))[0]
    for i in range(dim):
        cur_f = np.array([x[i] for x in features])
        # sorted(cur_f, key= lambda x: x[0])
        c_cs = []
        if threshes[i][0]:
            # discrete feature
            for t in threshes[i][1]:
                cur_r = cur_f - t
                cur_r = np.logical_xor(cur_r, n_ones)
                cur_r = np.logical_xor(cur_r, label_plus_one)
                # w_err = np.dot(cur_r, d)
                c_cs.append(cur_r)
        else:
            # continuous feature
            for t in threshes[i][1]:
                cur_r = np.sign(cur_f - t) + 1
                cur_r = np.logical_xor(cur_r, label_plus_one)
                # w_err = np.dot(cur_r, d)
                # n_err = np.dot(cur_r, n_ones)
                c_cs.append(cur_r)
        threshes_cheatsheet.append(c_cs)
    return threshes_cheatsheet



def pre_compute_threshes_3(features, label, threshes, d):
    '''

    :param features:
    :param label:
    :param threshes:
    :return:
    '''
    threshes_cheatsheet = []
    n, dim = np.shape(features)
    label_plus_one = np.array(label) + 1
    n_ones = np.ones((n, 1))
    for i in range(dim):
        cur_f = np.array([x[i] for x in features])

        # sorted(cur_f, key= lambda x: x[0])
        c_cs = []
        for t in threshes[i]:
            cur_r = np.sign(cur_f - t) + 1
            cur_r = np.logical_xor(cur_r, label_plus_one)
            w_err = np.dot(cur_r, d)
            n_err = np.dot(cur_r, n_ones)
            c_cs.append((w_err, n_err / n))
        threshes_cheatsheet.append(c_cs)
    return threshes_cheatsheet


def pre_compute_threshes_2(features, label, threshes, d):
    '''

    :param features:
    :param label:
    :param threshes:
    :return:
    '''
    threshes_cheatsheet = []
    n, dim = np.shape(features)
    for i in range(dim):
        cur_f = [(x[i], ind) for ind,x in enumerate(features)]
        sorted(cur_f, key= lambda x: x[0])
        cur_ind = 0
        c_cs = np.zeros((len(threshes[i]), 2)).tolist()
        for t in range(len(threshes[i])):
            if t > 0:
                c_cs[t][0] = c_cs[t-1][0]
                c_cs[t][1] = c_cs[t-1][1]
            while cur_f[cur_ind][0] <= threshes[i][t]:
                c_cs[t][0] += d[cur_f[cur_ind][1]]
                c_cs[t][1] += 1. / n
                cur_ind += 1
            w_err = 0.
            n_err = 0.
            for j in range(n):
                if features[j][i] > t:
                    if label[j] == -1:
                        w_err += d[i]
                        n_err += 1
                else:
                    if label[j] == 1:
                        w_err += d[i]
                        n_err += 1
            c_cs.append((w_err, n_err / n))
        threshes_cheatsheet.append(c_cs)
    return threshes_cheatsheet

def get_auc_from_predict(pred, label):
    unique_pred = np.unique(pred).tolist()
    unique_pred.sort(reverse=True)
    unique_pred = [max(unique_pred) + 1] + unique_pred
    # pred_label = [pl for pl in zip(list(pred), list(label))]
    # pred_label.sort(key=lambda pl:pl[0], reverse=True)
    # roc = [(0, 0)]
    roc = []
    n = len(pred)

    # d_predict = [y[0] for y in pred_label]
    # d_label = [y[1] for y in pred_label]
    pos = 0
    neg = 0
    for i in range(n):
        pos += 1 if label[i] == 1 else 0
        neg += 1 if label[i] == -1 else 0
    for t in unique_pred:
        roc.append(false_pos_true_pos(pred, label, pos, neg, t))

    # TODO calculate auc from roc
    return calculate_auc(roc)

def calculate_auc(roc):
    auc = 0
    for i in range(1, len(roc)):
        auc += (roc[i][0] - roc[i-1][0]) * (roc[i][1] + roc[i-1][1])
    return auc / 2.0

def false_pos_true_pos(pred, label, pos, neg, thresh):
    false_pos = 0
    true_pos = 0
    n = len(pred)
    for i in range(n):
        if pred[i] >= thresh:
            if label[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
    return (float(false_pos) / neg if neg > 0 else 0,
            float(true_pos) / pos if pos > 0 else 1)

def get_err_from_predict(pred, label):

    r_pred = np.sign(pred) + 1
    label_plus_one = np.array(label) + 1
    r = np.logical_xor(r_pred, label_plus_one).tolist()
    return sum(r) / len(label)

    # n = len(pred)
    # err = 0.
    # for i in range(n):
    #     if pred[i] >= 0 and label[i] == -1:
    #         err += 1
    #     elif pred[i] < 0 and label[i] == 1:
    #         err += 1
    # return err / n

def replace_zero_label_with_neg_one(data):
    data[1] = [-1 if l == 0 else l for l in data[1]]

def init_distribution(n):
    d = np.ones((1,n)) / n
    return d.tolist()[0]

def acc_stable(theta, features, label, last_n_accs, thresh):
    x = [[1] + f for f in features]
    x = np.array(x)
    y = np.dot(x, theta)
    y = [yy[0] for yy in y]
    label = [l[0] for l in label]
    cur_acc = acc(y, label)
    print('acc: ' + str(cur_acc))
    print('mse: ' + str(mse(y, label)))
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
            if predictions[i] > thresh:
                true_pos += 1
            else:
                false_neg += 1
        else:
            neg += 1
            if predictions[i] <= thresh:
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

def acc_exact(predictions, labels):
    n = len(labels)
    count = 0.
    for i in range(n):
        if predictions[i] == labels[i]:
            count += 1
    return count / n

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
    errs = np.array(predictions) - labels
    return sum(errs ** 2) / len(labels)
