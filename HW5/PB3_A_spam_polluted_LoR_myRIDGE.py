import DataLoader as loader
import Utilities as util
import Preprocess as prep
import GDModel as gd
import time

st = time.time()
# training parameter
result_path = 'results/PB3_A_spam_polluted_LoR.acc'
model_name = 'spam_'
train_data_path = 'data/spam_polluted/train/data.pickle'
test_data_path = 'data/spam_polluted/test/data.pickle'

# params
lamda = 0.0001
tol = 0.92
normalize_method = prep.zero_mean_unit_var
term_method = util.acc_higher_than_ridge

# laod and preprocess training data
tr_data = loader.load_pickle_file(train_data_path)
te_data = loader.load_pickle_file(test_data_path)
print('{:.2f} Data loaded!'.format(time.time() - st))

tr_data[0] = tr_data[0].tolist()
te_data[0] = te_data[0].tolist()

# normalize features
prep.normalize_features_all(normalize_method, tr_data[0], te_data[0])
print('{:.2f} Features normalized!'.format(time.time() - st))

is_batch = True
penalty = 'l2'  # l2 for RIDGE
alpha = 0.1
model = gd.LogisticRegressionGD(penalty, alpha)
model.build(tr_data[0], tr_data[1], lamda, term_method, tol, is_batch)
training_acc = model.test(tr_data[0], tr_data[1], util.acc)
testing_acc = model.test(te_data[0], te_data[1], util.acc)

print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, training_acc, testing_acc))

result = {}
result['TrainingAcc'] = training_acc
result['TestingAcc'] = testing_acc

# log the training result to file
util.write_result_to_file(result_path, model_name, result, True)