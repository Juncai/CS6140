from perceptron_dual import PerceptronDual
import csv
import Utilities as util
import numpy as np
import Consts as c
data_file = 'data/perceptronData.txt'

# load and preprocess data
features = []
labels = []
with open(data_file) as f:
    for line in csv.reader(f, delimiter='\t'):
        cur_l = int(line[-1])
        sign = 1
        cur_f = [sign * float(l) for l in line[:-1]]
        features.append(cur_f)
        labels.append([cur_l])
features = np.array(features)
labels = np.array(labels).transpose()[0]
# create perceptron
kernel = c.LINEAR
model = PerceptronDual(kernel=kernel)
model.fit(features, labels)

# calculate normalized weights
# theta = np.transpose(model.theta, (1, 0)).tolist()[0]
# norm_theta = [w / (-theta[0]) for w in theta[1:]]
# c_w_string = 'Classifier weights: '
# for w in theta:
#     c_w_string += str(w) + ' '
# n_w_string = 'Normalized with threshold: '
# for w in norm_theta:
#     n_w_string += str(w) + ' '
# print(c_w_string)
# print(n_w_string)

