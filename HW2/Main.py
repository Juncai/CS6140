import argparse
import DataLoader as loader
import Consts as c
import re
import Preprocess
import numpy as np
import ModelBuilder as builder
import Utilities
import os.path


def main(config_path):
    '''
    Main script for classifier building and testing
    '''
    config = loader.load_config(config_path)
    training_data = None
    testing_data = None
    # load training and testing data from files, normalize if necessary
    if c.TRAINING_D in config.keys():
        training_data = loader.load_dataset(config[c.TRAINING_D])
    if c.TESTING_D in config.keys():
        testing_data = loader.load_dataset(config[c.TESTING_D])
    if c.NORM_METHOD in config.keys():
        method = None
        if config[c.NORM_METHOD] == c.SHIFT_SCALE:
            method = Preprocess.shift_and_scale
        elif config[c.NORM_METHOD] == c.ZERO_MEAN_UNIT_VAR:
            method = Preprocess.zero_mean_unit_var
        if c.TESTING_D in config.keys():
            Preprocess.normalize_features_all(method, training_data[0], testing_data[0])
        else:
            Preprocess.normalize_features_all(method, training_data[0])

    # generate thresholds file if needed
    if c.THRESHS in config.keys() and not os.path.isfile(config[c.THRESHS]):
        Preprocess.generate_thresholds(training_data[0], config[c.THRESHS])

    # get path to store models and output results
    model_path = config[c.MODEL_PATH]
    output_path = config[c.OUTPUT_PATH]

    # use different validation method base on the config
    match = re.match(c.K_FOLD_RE, config[c.VALID_METHOD])
    if match:
        # perform k-fold validation
        k = int(match.group(c.K_GROUP))
        training_errs = []
        testing_errs = []
        for i in range(k):
            (tr_data, te_data) = Preprocess.prepare_k_fold_data(training_data, k, i + 1)
            model = builder.build_model(tr_data, config)
            training_errs.append(model.test(tr_data[0], tr_data[1], Utilities.get_test_method(config)))
            testing_errs.append(model.test(te_data[0], te_data[1], Utilities.get_test_method(config)))
        mean_training_err = np.mean(training_errs)
        mean_testing_err = np.mean(testing_errs)
        print str(k) + '-fold validation done. Training errors are:'
        print training_errs
        print 'Mean training error is:'
        print mean_training_err
        print 'Testing errors are:'
        print testing_errs
        print 'Mean testing error is:'
        print mean_testing_err
        config['TrainingErrs'] = str(training_errs)
        config['MeanTrainingErr'] = str(mean_training_err)
        config['TestingErrs'] = str(testing_errs)
        config['MeanTestingErr'] = str(mean_testing_err)
    elif config[c.VALID_METHOD] == c.HAS_TESTING_DATA:
        # perform testing with given testing dataset
        model = builder.build_model(training_data, config)
        training_err = model.test(training_data[0], training_data[1], Utilities.get_test_method(config))
        testing_err = model.test(testing_data[0], testing_data[1], Utilities.get_test_method(config))
        print 'Error for training data is:'
        print training_err
        print 'Error for testing data is:'
        print testing_err
        config['TrainingErr'] = str(training_err)
        config['TestingErr'] = str(testing_err)

    # Log the err
    f = open(output_path, 'w+')
    f.write(str(config))
    f.close()
    return


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-config', required=True)
    # opts = parser.parse_args()
    # main(opts.config)
    # main('spam.conf')
    # main('housing.conf')
    # main('housing_reg.conf')
    # main('spam_reg.conf')
    # main('housing_ridge.conf')
    # main('spam_ridge.conf')
    main('housing_linear_gd.conf')
