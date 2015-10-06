import argparse
import DataLoader as loader
import Consts as c
import re
import Preprocess
import numpy as np
import Utilities as util
import os.path
import Tree


# training parameter
k = 10
term_con = c.LAYER
term_thresh = 7
result_path = 'results/spamDT_9.acc'
model_name = 'spam_' + str(k) + 'fold_' + term_con + '_' + str(term_thresh)


# laod training data
training_data = loader.load_dataset('data/spambase.data')
# load threshold data
threshs = loader.load_pickle_file('config/spam_threshold')

# start training
training_accs = []
training_cms = []
testing_accs = []
testing_cms = []
for i in range(k):
    (tr_data, te_data) = Preprocess.prepare_k_fold_data(training_data, k, i + 1)


    tree = Tree.DecisionTree()
    tree.build(training_data[0],
               training_data[1], threshs, term_con, term_thresh)

    training_test_res = tree.test(tr_data[0], tr_data[1])
    training_accs.append(training_test_res[0])
    training_cms.append(training_test_res[1])
    testing_test_res = tree.test(te_data[0], te_data[1])
    testing_accs.append(testing_test_res[0])
    testing_cms.append(testing_test_res[1])

mean_training_acc = np.mean(training_accs)
mean_testing_acc = np.mean(testing_accs)
mean_training_cm = util.confusion_matrix_mean(training_cms)
mean_testing_cm = util.confusion_matrix_mean(testing_cms)

print str(k) + '-fold validation done. Training accs are:'
print training_accs
print 'Mean training acc is:'
print mean_training_acc
print 'Testing accs are:'
print testing_accs
print 'Mean testing acc is:'
print mean_testing_acc
print 'Mean Training Confusion Matrix is:'
print mean_training_cm
print 'Mean Testing Confusion Matrix is:'
print mean_testing_cm

result = {}
result['Termination condition'] = term_con
result['Termination threshold'] = term_thresh
result['Fold'] = str(k)
result['TrainingAccs'] = str(training_accs)
result['MeanTrainingAcc'] = str(mean_training_acc)
result['TestingAccs'] = str(testing_accs)
result['MeanTestingAcc'] = str(mean_testing_acc)
result['TrainingCMs'] = str(training_cms)
result['TestingCMs'] = str(testing_cms)
result['MeanTrainingCM'] = str(mean_training_cm)
result['MeanTestingCM'] = str(mean_testing_cm)


# log the training result to file
util.write_result_to_file(result_path, model_name, result)
