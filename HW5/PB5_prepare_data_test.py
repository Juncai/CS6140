from unittest import TestCase
import PB5_prepare_data as pd
import DataLoader as loader

class TestPB5_prepare_data(TestCase):
    def test_random_select_rectangle(self):
        h = 28
        w = 28
        pl = 130
        ph = 170
        n = 5
        act_res = pd.random_select_rectangle(h, w, n, pl, ph)
        self.assertEqual(len(act_res), 5)

    def test_compute_feature_with_cs(self):
        # pd.compute_feature_with_cs(((0, 0), (5, 6)), None)

        pass

    def test_count_black(self):
        rect = ((1, 1), (4, 5))
        cs = []
        css = loader.load_pickle_file('data/digits/')
        pass
