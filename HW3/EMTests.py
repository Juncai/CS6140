import unittest
import EM as em
import numpy as np


class EMTests(unittest.TestCase):

    # def test_logistic_fun(self):
    #     lrgd = gd.LogisticRegressionGD()
    #     theta = np.zeros((4, 1))
    #     features = [[2, 3, 4]]
    #     res = lrgd.logistic_fun(theta, features)
    #     self.assertEqual(res[0][0], 0.5)

    def test_mt_dot(self):
        m = np.matrix([[1, 2, 3], [4, 5, 6]])
        exp_res = [17, 29,45]
        act_res = em.my_dot(m)
        print act_res
        self.assertEqual(act_res, exp_res)




if __name__ == '__main__':
    unittest.main()
