import unittest
import Models
import math


class ModelsTests(unittest.TestCase):

    # def test_logistic_fun(self):
    #     lrgd = gd.LogisticRegressionGD()
    #     theta = np.zeros((4, 1))
    #     features = [[2, 3, 4]]
    #     res = lrgd.logistic_fun(theta, features)
    #     self.assertEqual(res[0][0], 0.5)

    def test_get_est(self):
        means = [1]
        model = Models.NBBernoulli(means)
        act_est = model._get_laplace_est(0, 2)
        exp_est = 0.25
        self.assertEqual(act_est, exp_est)

    def test_get_p_x_y(self):
        means = [1, 0.5]
        model = Models.NBBernoulli(means)
        model.px = [[[0.2, 0.8], [0.4, 0.6]],
                    [[0.3, 0.7], [0.5, 0.5]]]
        act_res = model._get_p_x_y(0, 0.6, 1)
        exp_res = 0.3
        self.assertEqual(act_res, exp_res)


    def test_NBB_build(self):
        means = [1, 0.5]
        model = Models.NBBernoulli(means)
        f = [[0.3, 1], [1.7, 0]]
        l = [1, 0]
        model.build(f, l)
        act_res = model.px
        exp_res = [[[1. / 3, 2. / 3], [2. / 3, 1. / 3]],
                   [[2. / 3, 1. / 3], [1. / 3, 2. / 3]]]
        self.assertEqual(act_res, exp_res)

    def test_gaussian_calculation(self):
        model = Models.NBGaussian()
        model.params = [[[0.5, 2]], [[1, 3]]]
        act_res = model._get_p_x_y(0, 0.5, 0)
        exp_res = 0.5 * math.pow(2 * math.pi, -0.5)
        self.assertEqual(act_res, exp_res)



if __name__ == '__main__':
    unittest.main()
