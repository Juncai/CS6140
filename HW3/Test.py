import GDModel as gd
import unittest
import numpy as np
import Utilities as util

class MyTests(unittest.TestCase):

    # def test_logistic_fun(self):
    #     lrgd = gd.LogisticRegressionGD()
    #     theta = np.zeros((4, 1))
    #     features = [[2, 3, 4]]
    #     res = lrgd.logistic_fun(theta, features)
    #     self.assertEqual(res[0][0], 0.5)

    def test_sigmoid(self):
        x_1 = 1
        x_0 = 0
        print util.sigmoid(x_1)
        self.assertEqual(util.sigmoid(x_0), 0.5)

    def test_nn_term_fun(self):
        output = [[0.9, 0.03, 0.2], [0.4, 0.92, 0.23]]
        exp = [[1, 0, 0], [0, 1, 0]]
        res = util.nn_term_fun(output, exp)
        self.assertEqual(res, 0.1351)


if __name__ == '__main__':
    unittest.main()
