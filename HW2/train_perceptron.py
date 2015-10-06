import NNModel
import csv
import Utilities as util
import argparse
import numpy as np

data_file = 'data/perceptronData.txt'
# parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('datafile', required=True, nargs=1)
# opts = parser.parse_args()
# data_file = opts['datafile']

# load and preprocess data
features = []
labels = []
with open(data_file) as f:
    for line in csv.reader(f, delimiter='\t'):
        cur_l = int(line[-1])
        # reflection
        sign = 1
        if cur_l == -1:
            sign = -1
            cur_l = -cur_l
        cur_f = [sign * float(l) for l in line[:-1]]
        features.append(cur_f)
        labels.append([cur_l])

# create perceptron
model = NNModel.Perceptron()
lamda = 0.01
model.build(features, labels, lamda, util.mistakes_less_than, 1)
# calculate normalized weights
theta = np.transpose(model.theta, (1, 0)).tolist()[0]
norm_theta = [w / (-theta[0]) for w in theta[1:]]
c_w_string = 'Classifier weights: '
for w in theta:
    c_w_string += str(w) + ' '
n_w_string = 'Normalized with threshold: '
for w in norm_theta:
    n_w_string += str(w) + ' '
print c_w_string
print n_w_string

