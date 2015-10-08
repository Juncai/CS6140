import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Consts as c
import RegressionModel as rm

def main():
    # training parameter
    result_path = 'results/housingRidge_1.mse'
    model_name = 'housing_shiftAndScale'
    # normalization = Preprocess.zero_mean_unit_var
    normalization = Preprocess.shift_and_scale
    cols_not_norm = (0,7)
    lamda = 1

    # laod and preprocess training data
    training_data = loader.load_dataset('data/housing_train.txt')
    testing_data = loader.load_dataset('data/housing_test.txt')
    Preprocess.normalize_features_all(normalization, training_data[0], testing_data[0], cols_not_norm)


    # start training
    model = rm.Ridge()
    model.build(training_data[0], training_data[1], lamda)
    training_mse = model.test(training_data[0], training_data[1], util.mse)
    testing_mse = model.test(testing_data[0], testing_data[1], util.mse)
    print 'Error for training data is:'
    print training_mse
    print 'Error for testing data is:'
    print testing_mse

    result = {}
    result['TrainingMSE'] = str(training_mse)
    result['TestingMSE'] = str(testing_mse)
    result['Theta'] = str(model.theta)

    # log the training result to file
    util.write_result_to_file(result_path, model_name, result)

if __name__ == '__main__':
    main()