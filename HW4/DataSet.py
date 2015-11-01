


class DataSet():
    data = []
    tr = []
    te = []
    re = []

    def __init__(self, datapoints):
        self.data = datapoints

    def random_pick(self, c):