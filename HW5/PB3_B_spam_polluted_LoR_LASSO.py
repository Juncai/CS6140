import DataLoader as loader
import Utilities as util
import Preprocess as prep
import time
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search

st = time.time()
# training parameter
result_path = 'results/PB3_B_spam_polluted_LoR_Lasso_sklearn.acc'
model_name = 'spam_'
train_data_path = 'data/spam_polluted/train/data.pickle'
test_data_path = 'data/spam_polluted/test/data.pickle'

# params
normalize_method = prep.zero_mean_unit_var
term_method = util.acc_higher_than

# laod and preprocess training data
tr_data = loader.load_pickle_file(train_data_path)
te_data = loader.load_pickle_file(test_data_path)
print('{:.2f} Data loaded!'.format(time.time() - st))

tr_data[0] = tr_data[0].tolist()
te_data[0] = te_data[0].tolist()

# normalize features
prep.normalize_features_all(normalize_method, tr_data[0], te_data[0])
print('{:.2f} Features normalized!'.format(time.time() - st))


# using sklearn with grid search
parameters = {'C' : [0.05], 'penalty' : ('l1',), 'tol' : (0.07,)}
model = LogisticRegression(C=0.05, penalty='l1', tol=0.08)
clf = grid_search.GridSearchCV(model, parameters)
clf.fit(tr_data[0], tr_data[1])

# model.fit(tr_data[0], tr_data[1])
# tr_pred = model.predict(tr_data[0])
# te_pred = model.predict(te_data[0])
tr_pred = clf.predict(tr_data[0])
te_pred = clf.predict(te_data[0])

print('Params: {}'.format(clf.get_params()))

training_acc = util.acc(tr_pred, tr_data[1])
testing_acc = util.acc(te_pred, te_data[1])

# old regression
# is_batch = True
# model = gd.LogisticRegressionGD()
# model.build(tr_data[0], tr_data[1], lamda, term_method, tol, is_batch)
# training_acc = model.test(tr_data[0], tr_data[1], util.acc)
# testing_acc = model.test(te_data[0], te_data[1], util.acc)

print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, training_acc, testing_acc))

result = {}
result['TrainingAcc'] = training_acc
result['TestingAcc'] = testing_acc

# log the training result to file
util.write_result_to_file(result_path, model_name, result, True)