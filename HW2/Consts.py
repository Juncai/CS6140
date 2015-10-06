import string

DATA_PATH = '/Users/juncai/Dropbox/CS6140/data/'
# Classifier types
REGRESSION = 'LinearRegression'
RIDGE = 'Ridge'
LINEAR_R_GD = 'LinearRGD'
LOGISTIC_R_GD = 'LogisticRGD'
PERCEPTRON = 'Perceptron'

# Config section name
SECTION = 'Classifier'

# Config item names
CLSFR_TYPE = string.lower('ClassifierType')
TRAINING_D = string.lower('TrainingData')
TESTING_D = string.lower('TestingData')
THRESHS = string.lower('FeatureThreshs')
TERM_CON = string.lower('TerminatingCondition')
TERM_THRESH = string.lower('TerminatingThreshold')
VALID_METHOD = string.lower('ValidationMethod')
MODEL_PATH = string.lower('ModelFile')
OUTPUT_PATH = string.lower('TestResult')
NORM_METHOD = string.lower('NormalizeMethod')
TEST_METHOD = string.lower('TestMethod')
LAMBDA = string.lower('Lambda')
UPDATE_STYLE = string.lower('UpdateStyle')

# Config item values
HAS_TESTING_DATA = 'TestingData'
SHIFT_SCALE = 'ShiftAndScale'
ZERO_MEAN_UNIT_VAR = 'ZeroMeanUnitVar'
LAYER = 'layer'
DATAPOINT = 'datapoint'
MSE = 'MSE'
ITERATION = 'Iteration'
ACC = 'Accuracy'
ERROR_DEC = 'errorDec'
MSE_TEST = 'MSE'
ACC_TEST = 'ACC'
BATCH = 'Batch'
STOCHASTIC = 'Stochastic'


# REs
K_GROUP = 'k'
K_FOLD_RE = '(?P<' + K_GROUP + '>\d+)-fold'