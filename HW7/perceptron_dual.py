import numpy as np
import Kernels
import Consts as c

class PerceptronDual():

    def __init__(self, kernel_fun=c.LINEAR, max_iter=None):
        self.m = []
        self.kernel = None
        self.features = []
        self.label = []
        self.kernel_func = Kernels.Kernels(kernel_fun, is_sim=True)
        self.predict_cache = []
        self.max_iter = max_iter
        pass

    def fit(self, features, label):
        self.features = features
        self.label = label
        self.kernel = self.kernel_func.get_value_bak(features)
        # self.kernel = np.dot(features, features.transpose())
        self.m = np.zeros((features.shape[0],))

        # TODO repeat training process until no mistakes made
        iter_count = 1
        self.predict_cache = self.predict(None, self.kernel, raw=True)
        accs = 0
        while True:
            # sign_tmp = np.sign(self.predict_cache)
            sign_tmp = self.predict(None, self.kernel)
            mis_count = (sign_tmp != self.label).sum()
            acc = 1 - mis_count / len(label)
            accs += acc
            print('Iteration {}, acc: {:.3f}'.format(iter_count, acc))
            if mis_count == 0:
                break

            # for ind, pc in enumerate(self.predict_cache):
            for ind, pc in enumerate(sign_tmp):
                if pc * self.label[ind] <= 0:
                    self.m[ind] += self.label[ind]
                    # self.predict_cache += self.label[ind] * self.kernel[ind, :]

                    # self.m[ind] += 1
                    # self.predict_cache += self.kernel[ind, :]
            iter_count += 1
            if self.max_iter and iter_count == self.max_iter:
                break
        print("{} iterations done, average acc: {:.3f}".format(iter_count, accs / iter_count))


    def predict(self, features, kernel_table=None, raw=False):
        if kernel_table is None:
            cur_kernel = self.kernel_func.get_value(self.features, features)
        else:
            cur_kernel = kernel_table
        pred = np.dot(self.m, cur_kernel)
        if not raw:
            pred = np.sign(pred)
        return pred
