__author__ = 'Jon'

import DataLoader as loader
import numpy as np
import generate_thresholds as gt
tr_data_path = 'data/8newsgroup/train.trec/feature_matrix.txt'
te_data_path = 'data/8newsgroup/test.trec/feature_matrix.txt'
tr_data_save_path = 'data/8newsgroup/train.data'
te_data_save_path = 'data/8newsgroup/test.data'
thresh_path = 'data/8newsgroup/8newsgroup.thresh'

# load data
tr_data = loader.load_8news_data(tr_data_path, tr_data_save_path)
te_data = loader.load_8news_data(te_data_path, te_data_save_path)

all_features = tr_data[0] + te_data[0]

# generate thresholds
gt.generate_thresholds_8news(all_features, thresh_path)

