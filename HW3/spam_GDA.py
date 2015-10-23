import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Models as m

# training parameter
k = 10  # fold
result_path = 'results/spamGDA_2.acc'
model_name = 'spam_' + str(k) + 'fold'


# laod and preprocess training data
training_data = loader.load_dataset('data/spambase.data')

# start training
training_accs = []
testing_accs = []

k_folds = Preprocess.prepare_k_folds(training_data, k)

for i in range(k):
    tr_data, te_data = Preprocess.get_i_fold(k_folds, i)

    model = m.GDA()
    model.build(tr_data[0], tr_data[1])

    training_test_res = model.test(tr_data[0], tr_data[1], util.acc)
    training_accs.append(training_test_res)
    testing_test_res = model.test(te_data[0], te_data[1], util.acc)
    testing_accs.append(testing_test_res)

mean_training_acc = np.mean(training_accs)
mean_testing_acc = np.mean(testing_accs)
print str(k) + '-fold validation done. Training accs are:'
print training_accs
print 'Mean training acc is:'
print mean_training_acc
print 'Testing accs are:'
print testing_accs
print 'Mean testing acc is:'
print mean_testing_acc

result = {}
result['Fold'] = str(k)
result['TrainingAccs'] = str(training_accs)
result['MeanTrainingAcc'] = str(mean_training_acc)
result['TestingAccs'] = str(testing_accs)
result['MeanTestingAcc'] = str(mean_testing_acc)
result['mu_0'] = str(model.mu[0])
result['mu_1'] = str(model.mu[1])
result['sigma'] = str(model.sigma)

# log the training result to file
util.write_result_to_file(result_path, model_name, result)
