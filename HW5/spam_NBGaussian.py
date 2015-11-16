import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Model as m

# training parameter
k = 10  # fold
result_path = 'results/spamNBGaussian_1.acc'
model_name = 'spam_' + str(k) + 'fold'



# laod and preprocess training data
training_data = loader.load_dataset('data/spambase.data')

# start training
training_accs = []
training_cms = []
testing_accs = []
testing_cms = []
roc = []
auc = 0.0
k_folds = Preprocess.prepare_k_folds(training_data, k)

for i in range(k):
    tr_data, te_data = Preprocess.get_i_fold(k_folds, i)


    model = m.NBGaussian()
    model.build(tr_data[0], tr_data[1])

    training_test_res = model.test(tr_data[0], tr_data[1], util.compute_acc_confusion_matrix)
    training_accs.append(training_test_res[0])
    training_cms.append(training_test_res[1])
    testing_test_res = model.test(te_data[0], te_data[1], util.compute_acc_confusion_matrix)
    testing_accs.append(testing_test_res[0])
    testing_cms.append(testing_test_res[1])

    # calculate ROC on fold 1
    if i == 1:
        roc = model.calculate_roc(training_data[0], training_data[1])
        auc = util.calculate_auc(roc)



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
print 'AOC for fold 1 is:'
print auc

result = {}
result['Fold'] = str(k)
result['TrainingAccs'] = str(training_accs)
result['MeanTrainingAcc'] = str(mean_training_acc)
result['TestingAccs'] = str(testing_accs)
result['MeanTestingAcc'] = str(mean_testing_acc)

result['TrainingCMs'] = str(training_cms)
result['TestingCMs'] = str(testing_cms)
result['MeanTrainingCM'] = str(mean_training_cm)
result['MeanTestingCM'] = str(mean_testing_cm)
result['ROC'] = str(roc)
result['AUC'] = str(auc)



# log the training result to file
util.write_result_to_file(result_path, model_name, result)
