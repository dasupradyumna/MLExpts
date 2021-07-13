from collections import Counter

import numpy as np


class KNearestNeighbour :

    def __init__( self, K, Dataset, DistanceMetric ) :
        self.k = K
        self.datapoints = Dataset[0]  # 2d array, dim0 : no. of datapoints, dim1 : data contained in each point
        self.labels = Dataset[1]  # 1d array of labels for each datapoint
        self.distance_metric = DistanceMetric

    # test point is 1d array
    def __call__( self, test_point ) :
        norms = self.distance_metric(test_point, self.datapoints)
        k_nearest_idx = np.argpartition(norms, self.k - 1)[:self.k]
        k_nearest_labels = self.labels[k_nearest_idx]
        return Counter(k_nearest_labels).most_common(1)[0][0]


class Linear :

    def __init__( self, NumClasses, LossMetric ) :
        self.weights = None
        self.num_classes = NumClasses
        self.loss = LossMetric

    def train( self, datapoints, labels, learning_rate, num_iterations ) :
        pass

    def predict( self, test_point ) :
        pass
