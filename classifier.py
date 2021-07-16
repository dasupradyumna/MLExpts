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

    def __init__( self, NumClasses ) :
        self.weights = None
        self.num_classes = NumClasses
        self.loss_model = None

    def setLoss( self, LossModel ) :
        self.loss_model = LossModel

    def train( self, datapoints, labels, num_iterations, learning_rate ) :
        if not self.weights :
            self.weights = 1e-3 * np.random.randn(datapoints.shape[1], self.num_classes)

        labels = np.eye(self.num_classes, dtype=bool)[labels]  # convert label values to one-hot vectors

        batch_size = datapoints.shape[0] // num_iterations
        loss_iterations = np.zeros(num_iterations)
        for itr in np.random.permutation(range(num_iterations)) :
            data_batch = datapoints[itr * batch_size : (itr + 1) * batch_size]
            labels_batch = labels[itr * batch_size : (itr + 1) * batch_size]

            loss = self.loss_model.evaluate(self.weights, data_batch, labels_batch)
            gradients = self.loss_model.gradient(self.weights, data_batch, labels_batch)
            # import metrics
            # num_gradients = metrics.gradient_check(self.weights, data_batch, labels_batch, self.loss_model)
            loss_iterations[itr] = loss
            self.weights -= learning_rate * gradients

        return loss_iterations

    def predict( self, test_points ) :  # test_points can be 1d or 2d array
        axis = int(test_points.ndim == 2)
        return np.argmax(test_points @ self.weights, axis)
