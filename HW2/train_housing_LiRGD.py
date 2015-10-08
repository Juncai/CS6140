import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Consts as c
import GDModel as gd

# training parameter
result_path = 'results/housingLiRGD_1.mse'
model_name = 'housing'
lamda = 0.0001  # 0.000015
is_batch = False
# normalization = Preprocess.zero_mean_unit_var
normalization = Preprocess.shift_and_scale
term_fun = util.mse_less_than
term_thresh = 22.17
cols_not_norm = [0,7]

# laod and preprocess training data
training_data = loader.load_dataset('data/housing_train.txt')
testing_data = loader.load_dataset('data/housing_test.txt')
Preprocess.normalize_features_all(normalization, training_data[0], testing_data[0], not_norm=cols_not_norm)



# start training
model = gd.LinearRegressionGD()
model.build(training_data[0], training_data[1], lamda, term_fun, term_thresh, is_batch)
try:
    pass
except KeyboardInterrupt:
    print 'Interrupted'
finally:
    training_mse = model.test(training_data[0], training_data[1], util.mse)
    testing_mse = model.test(testing_data[0], testing_data[1], util.mse)
    print 'Error for training data is:'
    print training_mse
    print 'Error for testing data is:'
    print testing_mse

result = {}
result['TrainingMSE'] = str(training_mse)
result['TestingMSE'] = str(testing_mse)
result['Iteration'] = str(model.iter_count)

# log the training result to file
util.write_result_to_file(result_path, model_name, result)
