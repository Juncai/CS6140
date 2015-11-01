import DataLoader as loader
import Consts as c
import Utilities as utils
import Tree
import RegressionModel as rmodel
import DataLoader as loader


def build_model(training_data, config):
    '''
    Build model from the config and training data
    '''
    m_type = config[c.CLSFR_TYPE]
    if m_type == c.DT_WITH_IG:
        # for decision tree
        # load thresholds
        threshs = loader.load_arrays(config[c.THRESHS])

        tree = Tree.Tree()
        tree.build(utils.split_on_ig, training_data[0],
                   training_data[1], threshs, config[c.TERM_CON], int(config[c.TERM_THRESH]))
        return tree
    elif m_type == c.REGRESSION_TREE:
        # for regression tree
        # load thresholds
        threshs = loader.load_arrays(config[c.THRESHS])

        tree = Tree.Tree()
        tree.build(utils.split_on_mse, training_data[0],
                  training_data[1], threshs, config[c.TERM_CON], float(config[c.TERM_THRESH]))
        return tree
    elif m_type == c.REGRESSION:
        # for linear regression
        reg_model = rmodel.Regression()
        reg_model.build(training_data[0], training_data[1])
        return reg_model
