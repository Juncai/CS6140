import DecisionTree as dt


class Bagging():

    def __init__(self):
        self.trees = []


    def test(self, features, labels, err_fun):
        return err_fun(self.predict(features), labels)

    def test_with_model(self, m_ind, features, labels, err_fun):
        m = self.trees[m_ind]
        pred = []
        for f in features:
            pred.append(m.predict(f))
        return err_fun(pred, labels)

    def predict(self, features):
        preds = []
        for f in features:
            preds.append(self.predict_single(f))
        return preds


    def predict_single(self, feature):
        T = len(self.trees)
        res = 0.
        for i in range(T):
            pred = self.trees[i].predict(feature)
            res += pred
        return res / T


    def add_tree(self, features, label, threshes, layer_thresh, model_class=None):
        if model_class is None:
            t = dt.DecisionTree()
            t.build(features, label, threshes, layer_thresh)
            self.trees.append(t)
        else:
            t = model_class(max_depth=3)
            t.fit(features, label)
            self.trees.append(t)
