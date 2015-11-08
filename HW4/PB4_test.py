__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import Utilities as util
import DecisionStump as ds
import Boosting as b
import pickle
import math
from scipy.spatial.distance import hamming
import os.path as path
import copy
import time

# training parameter
ecoc_path = 'data/8newsgroup/ecoc_cs'


ecoc_cs = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
           [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
           [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1]]

ecoc_cs_t = np.transpose(ecoc_cs).tolist()

ecoc_to_store = [0, ecoc_cs_t, ecoc_cs]
with open(ecoc_path, 'wb+') as f:
    pickle.dump(ecoc_to_store, f)

print('ECOC generation done!')




