from perceptron_dual import PerceptronDual
import csv
import Utilities as util
import numpy as np
import Consts as c
import Preprocess


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
Preprocess.normalize_features_all(Preprocess.zero_mean_unit_var, features)
labels = np.array(labels).transpose()[0]

# create perceptron
kernel = c.LINEAR
# kernel = c.GAUSSIAN
# kernel = c.EUCLIDEAN
# kernel = c.POLY
model = PerceptronDual(kernel_fun=kernel)
model.fit(features, labels)

