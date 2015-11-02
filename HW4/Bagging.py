import numpy as np
import math
import DecisionTree as dt


class Bagging():

    def __init__(self):
        self.trees = []


    def test(self, features, labels, err_fun):
        return err_fun(self.predict(features), labels)


    def predict(self, features):
        preds = []
        for f in features:
            preds.append(self.predict_single(f))
        return preds


    def predict_single(self, feature):
        T = len(self.trees)
        res = 0.
        for i in range(T):
            pred = self.trees[i].single_predict(feature)
            res += pred
        return res / T


    def add_tree(self, features, label, threshes, layer_thresh):
        t = dt.DecisionTree()
        t.build(features, label, threshes, layer_thresh)
        self.trees.append(t)