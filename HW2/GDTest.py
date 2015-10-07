import GDModel as gd
import Utilities as util

features = [[1, 1, 1],
            [1, 0, 1],
            [0, 1, 0]]

label = [1, 0, 0]


lamda = 0.1
thresh = 0.001
is_batch = False

model = gd.LinearRegressionGD()
model.build(features, label, lamda, util.mse_less_than, thresh, is_batch)

