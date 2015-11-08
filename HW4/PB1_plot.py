from matplotlib import pyplot as plt
import DataLoader as loader
import numpy as np
# result = {}
# result['Fold'] = k
# result['Trainingerrs'] = training_errs
# result['MeanTrainingAcc'] = mean_training_err
# result['Testingerrs'] = testing_errs
# result['MeanTestingAcc'] = mean_testing_err
# result['1stBoostTrainingError'] = tr_errs_1st_boost
# result['1stBoostTestingError'] = te_errs_1st_boost
# result['1stBoostModelError'] = round_err_1st_boost
# result['1stBoostTestingAUC'] = te_auc_1st_boost
# result['1stBoostTestingROC'] = te_roc_1st_boost

# DS_TYPE = 'Random'
DS_TYPE = 'Optimal'

if DS_TYPE == 'Random':
    result_path = 'results/spamRDSBoosting_final.acc.pickle'
else:
    result_path = 'results/spamODSBoosting_final.acc.pickle'

# target = 'auc'
# target = 'errs'
target = 'm_err'

result = loader.load_pickle_file(result_path)
n_round = len(result['1stBoostTestingAUC'])


if target == 'auc':
    auc = result['1stBoostTestingAUC']
    x = [i+1 for i in range(n_round)]

    plt.plot(x, auc, color='red', linestyle='solid')
    plt.title("Adaboost with " + DS_TYPE + "DecisionStump - AUC")
    plt.xlabel("Iteration Step")
    plt.ylabel("AUC")
    plt.show()

if target == 'errs':
    tr_err = result['1stBoostTrainingError']
    te_err = result['1stBoostTestingError']

    x = [i+1 for i in range(n_round)]

    plt.plot(x, tr_err, color='red', linestyle='solid')
    plt.plot(x, te_err, color='blue', linestyle='solid')
    plt.title("Adaboost with " + DS_TYPE + "DecisionStump - Train/Test Error")
    plt.xlabel("Iteration Step")
    plt.ylabel("Error")
    plt.show()

if target == 'm_err':
    m_err = result['1stBoostModelError']
    x = [i+1 for i in range(n_round)]

    plt.plot(x, m_err, color='red', linestyle='solid')
    plt.title("Adaboost with " + DS_TYPE + "DecisionStump - Round Error")
    plt.xlabel("Iteration Step")
    plt.ylabel("Error")
    plt.show()
