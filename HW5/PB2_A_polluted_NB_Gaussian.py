import DataLoader as loader
import Preprocess
import numpy as np
import Utilities as util
import Model as m
import profile
import time


def main():
    st = time.time()
    # training parameter
    result_path = 'results/PB2_A_spam_polluted_NB_Gaussian.acc'
    model_name = 'spam_'
    train_data_path = 'data/spam_polluted/train/data.pickle'
    test_data_path = 'data/spam_polluted/test/data.pickle'

    # laod and preprocess training data
    tr_data = loader.load_pickle_file(train_data_path)
    te_data = loader.load_pickle_file(test_data_path)
    print('{:.2f} Data loaded!'.format(time.time() - st))

    # start training
    model = m.NBGaussian()
    model.build(tr_data[0], tr_data[1])

    training_acc = model.test(tr_data[0], tr_data[1], util.acc)
    # training_cms.append(training_test_res[1])
    testing_acc = model.test(te_data[0], te_data[1], util.acc)
    # testing_cms.append(testing_test_res[1])


    print('{} Final results. Train acc: {}, Test acc: {}'.format(time.time() - st, training_acc, testing_acc))

    result = {}
    result['TrainingAcc'] = training_acc
    result['TestingAcc'] = testing_acc

    # log the training result to file
    util.write_result_to_file(result_path, model_name, result, True)

if __name__ == '__main__':
    # profile.run('main()')
    main()