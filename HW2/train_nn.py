import NNModel
import Utilities as util

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
lamda = 3
try:
    model.build((nn_input, nn_output), lamda, util.mse_for_nn, 1000)
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


