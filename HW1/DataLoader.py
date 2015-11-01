'''
main module for the task.
'''

import ConfigParser
import Consts
import numpy as np

classifier_type = ''
training_data = ()
testing_data = None
feature_threshold = []
norm_req = []
model_path = ''
output_path = ''


def load_arrays(path):
    '''
    Read other data from file, no need to be matrix
    '''
    f = open(path, 'r')
    ar = []
    for line in f:
        ar.append([float(x) for x in line.split(',')])
    return ar


def load_dataset(path, has_label=True):
    '''
    Read data from a file and store them in a matrix
    '''
    f = open(path, 'r')
    deli = ','
    first_line = f.readline()
    if len(first_line.split(None)) > 1:
        # only works for more than one feature!
        deli = None
    f.close()
    data_set = np.genfromtxt(path, delimiter=deli)

    if has_label:
        label = [x[-1] for x in data_set]
        features = [x[:-1] for x in data_set]
        return [features, label]
    else:
        return (data_set, [])


def load_config(path):
    config_dict = {}
    config = ConfigParser.ConfigParser()
    config.read(path)
    for item in config.items(Consts.SECTION):
        config_dict[item[0]] = item[1]
    return config_dict

# def load_data_from_config(path):
#     '''
#     Need to run this method first!!
#     '''
#     global classifier_type, training_data, testing_data
#     global feature_threshold, norm_req
#     global model_path, output_path
#     config = ConfigParser.ConfigParser({'TestingData' : None})
#     config.read(path)
#     classifier_type = config.get(Consts.SECTION, 'ClassifierType', 0)
#     training_data_path = config.get(Consts.SECTION, 'TrainingData', 0)
#     training_data = Utilities.file_to_dataset(training_data_path)
#     testing_data_path = config.get(Consts.SECTION, 'TestingData', 0)
#     if testing_data_path:
#         testing_data = Utilities.file_to_dataset(testing_data_path)
#     model_path = config.get(Consts.SECTION, 'ModelFile', 0)
#     output_path = config.get(Consts.SECTION, 'TestResult', 0)

#     if classifier_type in (Consts.DT_WITH_IG, Consts.REGRESSION_TREE):
#         feature_threshold_path = config.get(Consts.SECTION, 'FeatureThreshs', 0)
#         feature_threshold = Utilities.file_to_array(feature_threshold_path)
#     elif classifier_type == Consts.REGRESSION:
#         norm_req_path = config.get(Consts.SECTION, 'NormalizeRequirements', 0)
#         norm_req = Utilities.file_to_array(norm_req_path)




# if __name__ == '__main__':
#     load_data_from_config('spam.conf')
#     build_tree()
#     err = test_tree_from_file(model_path, training_data)
#     print err
    # f = open(output_path, 'w+')
    # f.write(err)
    # f.close()
