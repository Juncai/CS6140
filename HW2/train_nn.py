import NNModel
import csv
import Utilities as util
import argparse
import numpy as np

# init training data
nn_input = [[1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]]

nn_output = nn_input

# create perceptron
model = NNModel.NN()
lamda = 0.1
try:
    model.build((nn_input, nn_output), lamda, util.mse_for_nn, 0.2)
except KeyboardInterrupt:
    pass
# calculate normalized weights
print 'Final results:'
print model.w_in
print model.w_out
print model.y
print model.z
print model.iter_count
print model.mse


