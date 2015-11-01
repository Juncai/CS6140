import string
# Classifier types
DT_WITH_IG = 'DT'
REGRESSION_TREE = 'RT'
REGRESSION = 'R'

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

# Config item values
HAS_TESTING_DATA = 'TestingData'
SHIFT_SCALE = 'ShiftAndScale'
ZERO_MEAN_UNIT_VAR = 'ZeroMeanUnitVar'
LAYER = 'layer'
DATAPOINT = 'datapoint'
ERROR_DEC = 'errorDec'

# REs
K_GROUP = 'k'
K_FOLD_RE = '(?P<' + K_GROUP + '>\d+)-fold'