import DataLoader as loader
import Consts as c
import Utilities as u
import RegressionModel as rmodel
import GDModel as gd


def build_model(training_data, config):
    '''
    Build model from the config and training data
    '''
    m_type = config[c.CLSFR_TYPE]
    if m_type == c.REGRESSION:
        # for linear regression
        reg_model = rmodel.LinearRegression()
        reg_model.build(training_data[0], training_data[1])
        return reg_model
    elif m_type == c.RIDGE:
        # for Ridge regression
        lamda = float(config[c.LAMBDA])
        model = rmodel.Ridge()
        model.build(training_data[0], training_data[1], lamda)
        return model
    elif m_type == c.LINEAR_R_GD:
        lamda = float(config[c.LAMBDA])
        term_fun = u.get_term_fun(config)
        thresh = float(config[c.TERM_THRESH])
        is_batch = u.get_is_batch(config)
        model = gd.LinearRegressionGD()
        model.build(training_data[0], training_data[1], lamda, term_fun, thresh, is_batch)
        return model


