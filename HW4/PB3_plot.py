from matplotlib import pyplot as plt
import DataLoader as loader
import numpy as np


# result['Trainingerrs'] = training_errs
# result['MeanTrainingAcc'] = mean_training_err
# result['Testingerrs'] = testing_errs
# result['MeanTestingAcc'] = mean_testing_err

# DS_TYPE = 'Random'

random_result_path = 'results/spamActive_random_final.acc.pickle'
optimal_result_path = 'results/spamActive_final.acc.pickle'

target = 'errs'

random_result = loader.load_pickle_file(random_result_path)
optimal_result = loader.load_pickle_file(optimal_result_path)

random_percent_list = [5, 10, 15, 20, 30, 50]
optimal_percent_list = [5 + 2 * i for i in range(23)]
optimal_percent_list.append(50)
# n_round = len(result['Testingerrs'])


if target == 'errs':
    # tr_err = result['1stBoostTrainingError']
    random_te_err = random_result['Testingerrs']
    optimal_te_err = optimal_result['Testingerrs']

    # x = [i+1 for i in range(n_round)]

    line_rand, = plt.plot(random_percent_list, random_te_err, color='black', linestyle='solid', label='Random')
    line_opt, = plt.plot(optimal_percent_list, optimal_te_err, color='green', linestyle='solid', label='Min')
    plt.legend(handles=[line_rand, line_opt])
    plt.title("Active Learning with DecisionStump - Test Error")
    plt.xlabel("% Label Data")
    plt.ylabel("Test Error")
    plt.show()

