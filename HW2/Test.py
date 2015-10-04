import GDModel as gd
import unittest
import numpy as np

class MyTests(unittest.TestCase):

    def test_logistic_fun(self):
        lrgd = gd.LogisticRegressionGD()
        theta = np.zeros((4, 1))
        features = [[2, 3, 4]]
        res = lrgd.logistic_fun(theta, features)
        self.assertEqual(res[0][0], 0.5)


if __name__ == '__main__':
    unittest.main()
