__author__ = 'Jon'

import DataLoader as loader
import generate_thresholds as gt
tr_data_path = 'data/8newsgroup/train.trec/feature_matrix.txt'
te_data_path = 'data/8newsgroup/test.trec/feature_matrix.txt'
tr_data_save_path = 'data/8newsgroup/train.data'
te_data_save_path = 'data/8newsgroup/test.data'
thresh_path = 'data/8newsgroup/8newsgroup.thresh'
thresh_path_v2 = 'data/8newsgroup/8newsgroup_f_i.thresh'

# load data
# tr_data = loader.load_8news_data(tr_data_path, tr_data_save_path)
# te_data = loader.load_8news_data(te_data_path, te_data_save_path)

# all_features = tr_data[0] + te_data[0]

# generate thresholds
# gt.generate_thresholds_8news(tr_data[0], thresh_path)




# re-generate thresholds
tr_data = loader.load_pickle_file(tr_data_save_path)
gt.generate_thresholds_8news_v2(tr_data[0], thresh_path_v2)
