import GDModel as gd
import Utilities as util

# features = [[1, 1, 1],
#             [1, 0, 1],
#             [0, 1, 0]]
#
# label = [1, 0, 0]
#
#
# lamda = 0.1
# thresh = 0.001
# is_batch = False
#
# model = gd.LogisticRegressionGD()
# model.build(features, label, lamda, util.mse_less_than, thresh, is_batch)


# features = [[1, 2, 3, 4],
#             [2, 2, 2, 2],
#             [4, 3, 2, 1],
#             [1, 1, 3, 2]]
#
# label = [38, 28, 32, 22]
#
# lamda = 0.0001
# thresh = 0.01
# is_batch = False
#
# model = gd.LinearRegressionGD()
# model.build(features, label, lamda, util.mse_less_than, thresh, is_batch)
# print model.theta

features = [[0.5, 0.1, 0.3, 0.6],
            [0.1, 0.8, 0.6, 0.2],
            [0.2, 0.5, 0.1, 0.4],
            [0.1, 0.7, 0.7, 0.1]]

label = [1, 0, 1, 0]

lamda = 0.0001
thresh = 0.9
is_batch = False

# model = gd.LinearRegressionGD()
# model.build(features, label, lamda, util.mse_less_than, thresh, is_batch)
model = gd.LogisticRegressionGD()
model.build(features, label, lamda, util.acc_higher_than, thresh, is_batch)
print model.theta