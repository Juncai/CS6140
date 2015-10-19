import numpy as np
import math


class Model():
    def test(self, features, labels, err_fun):
        return err_fun(self.predict(features), labels)

    def predict(self, features):
        '''

        :param features: a list of features as a list of lists
        :return: predictions made from the import features as a list of floats
        '''
        res = []
        for f in features:
            res.append(self.predict_single(f))
        return res

    def predict_single(self, feature):
        pass

    def calculate_roc(self, features, label):
        y_1_d = self.predict(features)
        # print y_1_d
        d = []
        for i in range(len(label)):
            d.append([y_1_d[i], label[i]])
        d.sort(key=lambda x: x[0],reverse=True)
        d_predict = [y[0] for y in d]
        d_label = [y[1] for y in d]
        pos = reduce(lambda x, y: x + 1 if y == 1 else x, [0] + d_label)
        neg = reduce(lambda x, y: x + 1 if y == 0 else x, [0] + d_label)
        roc = []
        for i in range(len(d)):
            roc.append(self.false_pos_true_pos(d_predict, d_label,
                                               pos, neg, i))
        return roc

    def false_pos_true_pos(self, pred, label, pos, neg, ind):
        false_pos = 0
        true_pos = 0
        for i in range(ind+1):
            if label[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        return (float(false_pos) / neg if neg > 0 else 0,
                float(true_pos) / pos if pos > 0 else 1)


class GDA(Model):
    mu = [[], []]
    mu_all = []
    sigma = []
    sigma_inv = []
    fi = None  # (fi_neg, fi_pos)
    coe = 0.0  # value of 1 / (2 * pi ^ (n / 2) * |sigma| ^ 0.5)


    def __init__(self):
        self.mu = [[], []]
        self.mu_all = []
        self.sigma = []
        self.sigma_inv = []
        self.fi = None
        self.coe = 0.0


    def build(self, features, labels):
        '''

        :param features: m*n matrix of features
        :param labels: 1*m array of labels
        :return:
        '''
        x = np.array(features)
        # get + and - count, calculate the fi
        m = len(labels)
        n = len(features[0])
        count_res = np.bincount(labels)
        neg_count = count_res[0]
        pos_count = count_res[1]
        fi_pos = 1.0 * pos_count / m
        self.fi = (1 - fi_pos, fi_pos)
        # TODO calculate mu_0, mu_1, mu_all
        for i in range(n):
            sum_0 = 0.0
            sum_1 = 0.0
            for j in range(m):
                if labels[j] == 0:
                    sum_0 += features[j][i]
                else:
                    sum_1 += features[j][i]
            self.mu[0].append(sum_0 / neg_count)
            self.mu[1].append(sum_1 / pos_count)
            self.mu_all.append((sum_0 + sum_1) / m)
        # TODO calculate sigma as a shared covariance matrix
        self.sigma = np.zeros((n, n))
        for i in range(m):
            x_minus_mu_all = np.matrix(x[i] - self.mu_all)  # 1*n
            x_minus_mu_all_t = np.transpose(x_minus_mu_all, (1, 0))
            tmp = np.dot(x_minus_mu_all_t, x_minus_mu_all)
            self.sigma += tmp
        self.sigma /= m
        self.sigma_inv = np.linalg.inv(self.sigma)
        # TODO calculate coe
        self.coe = self._compute_coe()

    def _compute_coe(self):
        return 1.0 / (math.pow(2 * math.pi, len(self.sigma) / 2.0) * math.pow(np.linalg.det(self.sigma), 0.5))

    def predict_single(self, feature):
        p_0 = self.compute_prob(feature, 0)
        p_1 = self.compute_prob(feature, 1)
        return 0 if p_0 > p_1 else 1

    def compute_prob(self, feature, label):
        x = np.array(feature)
        x_minus_mu = np.matrix(x - self.mu[label])
        tmp = np.dot(np.dot(x_minus_mu, self.sigma_inv), np.transpose(x_minus_mu, (1, 0)))
        return self.coe * math.exp(- 0.5 * tmp) * self.fi[label]


class NB(Model):
    py = []
    def _get_p_x_y(self, x_ind, x_val, y):
        pass

    def predict_single(self, feature):
        # init the possibilities
        p = [self.py[0], self.py[1]]
        # TODO calculate the possibility of label 1
        for y in (0, 1):
            for ind, x in enumerate(feature):
               p[y] *= self._get_p_x_y(ind, x, y)
        return 0 if p[0] > p[1] else 1

    def build(self, features, labels):
        # TODO calculate prior
        self.py = self._get_prior(labels)

        # TODO calculate p(x|y)
        self._build_model(features, labels)

    def _build_model(self, features, labels):
        pass

    def _get_prior(self, labels):
        count = len(labels)
        y_pos_cnt = 0.
        for y in labels:
            if y == 1.0:
                y_pos_cnt += 1
        return [y_pos_cnt / count, 1 - (y_pos_cnt / count)]

    def _get_laplace_est(self, cnt1, cnt2):
        '''
        with laplace smoothing
        :param cnt1:
        :param cnt2:
        :return:
        '''
        return 1.0 * (cnt1 + 1) / (cnt1 + cnt2 + 2)

class NBBernoulli(NB):
    px = []  # class based possibilities of the features
    means = []

    def __init__(self, means):
        self.means = means

    def _build_model(self, features, labels):
        f_len = len(features[0])
        count = len(features)
        self.px = [[], []]
        for i in range(f_len):
            pos_spam_cnt = 0
            pos_non_spam_cnt = 0
            neg_spam_cnt = 0
            neg_non_spam_cnt = 0

            for j in range(count):
                if features[j][i] > self.means[i]:
                    if labels[j] == 1:
                        pos_spam_cnt += 1
                    else:
                        pos_non_spam_cnt += 1
                else:
                    if labels[j] == 1:
                        neg_spam_cnt += 1
                    else:
                        neg_non_spam_cnt += 1
            self.px[0].append([self._get_laplace_est(neg_non_spam_cnt, pos_non_spam_cnt),
                               self._get_laplace_est(pos_non_spam_cnt, neg_non_spam_cnt)])
            self.px[1].append([self._get_laplace_est(neg_spam_cnt, pos_spam_cnt),
                               self._get_laplace_est(pos_spam_cnt, neg_spam_cnt)])

    def _get_p_x_y(self, x_ind, x_val, y):
        return self.px[int(y)][x_ind][0 if x_val <= self.means[x_ind] else 1]


class NBGaussian(NB):
    params = [[], []]

    def _build_model(self, features, labels):
        # TODO calculate conditional mean and variance for all the features
        f_len = len(features[0])
        count = len(features)
        self.params = [[], []]
        for i in range(f_len):

            spam_dp = []
            non_spam_dp = []

            for j in range(count):
                if labels[j] == 1:
                    spam_dp.append(features[j][i])
                else:
                    non_spam_dp.append(features[j][i])
            spam_e = np.mean(spam_dp)
            spam_std = np.std(spam_dp)
            non_spam_e = np.mean(non_spam_dp)
            non_spam_std = np.std(non_spam_dp)
            self.params[0].append([non_spam_e, non_spam_std if non_spam_std != 0 else 0.5])
            self.params[1].append([spam_e, spam_std if spam_std != 0 else 0.5])

    def _get_p_x_y(self, x_ind, x_val, y):
        e, std = self.params[int(y)][x_ind]
        # TODO calculate p
        res = 1 / math.pow(2 * math.pi * (std ** 2), 0.5) * math.pow(math.e, (-(x_val - e) ** 2) / (2 * std ** 2))
        return res


class NBHistogram(NB):
    histo = []

    def _build_model(self, features, labels):
        # TODO calculate conditional histogram for all the features
        f_len = len(features[0])
        count = len(features)
        self.histo = [[], []]

        for i in range(f_len):
            spam_dp = []
            non_spam_dp = []
            all_dp = [x[i] for x in features]
            for j in range(count):
                if labels[j] == 1:
                    spam_dp.append(features[j][i])
                else:
                    non_spam_dp.append(features[j][i])

            max = np.max(all_dp)
            min = np.min(all_dp)
            mean = np.mean(all_dp)
            spam_mean = np.mean(spam_dp)
            non_spam_mean = np.mean(non_spam_dp)

            spam_histo = self._get_histo(spam_dp, min, max, mean, spam_mean, non_spam_mean)
            non_spam_histo = self._get_histo(non_spam_dp, min, max, mean, spam_mean, non_spam_mean)

            self.histo[0].append(non_spam_histo)
            self.histo[1].append(spam_histo)


    def _get_histo(self, dp, min, mean, max, sub_mean_1, sub_mean_2):
        dp_len = len(dp)
        bins = []
        bins.append(min)
        if sub_mean_1 < sub_mean_2:
            bins.append(sub_mean_1)
            bins.append(mean)
            bins.append(sub_mean_2)
        else:
            bins.append(sub_mean_2)
            bins.append(mean)
            bins.append(sub_mean_1)
        bins.append(max)

        px = [0., 0., 0., 0.]
        for x in dp:
            for ind, val in enumerate(bins):
                if x <= val:
                    px[ind - 1] += 1

        px = [((p + 1) / (dp_len + 2)) for p in px]

        return bins, px

    def _get_p_x_y(self, x_ind, x_val, y):
        bins, px = self.histo[int(y)][x_ind]
        for ind, val in enumerate(bins):
            if val < 
        return res
