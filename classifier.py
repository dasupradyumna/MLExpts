from collections import Counter

import numpy as np

import metrics


# Represents a KNN classifier model
class KNearestNeighbour :

    def __init__( self, K, Dataset, DistanceMetric ) :
        self.k = K
        self.datapoints = Dataset[0]  # 2d array, dim0 : no. of datapoints, dim1 : data contained in each point
        self.labels = Dataset[1]  # 1d array of labels for each datapoint
        self.distance_metric = DistanceMetric

    # calling an object of this class will give the prediction for the argument datapoint
    # only pass a single datapoint, does not support prediction of a set of points
    def __call__( self, test_point ) :
        norms = self.distance_metric(self.datapoints - test_point)  # find distance to every point in training dataset
        k_nearest_idx = np.argpartition(norms, self.k - 1)[:self.k]  # partially sort k smallest values
        k_nearest_labels = self.labels[k_nearest_idx]  # find labels corresponding to above k smallest values
        return Counter(k_nearest_labels).most_common(1)[0][0]  # return prediction by majority count


# Represent a simple Linear (affine) classifier model
class Linear :

    def __init__( self, NumClasses ) :
        self.weights = None
        self.num_classes = NumClasses
        self.loss_model = None

    # change the loss metric being used for training
    def setLoss( self, LossModel ) :
        self.loss_model = LossModel

    # train the weights of the model using a dataset and other parameters
    def train( self, datapoints, labels, num_iterations, learning_rate ) :
        if not self.weights :  # random initialize (Gaussian) if weight matrix is uninitialized
            self.weights = 1e-2 * np.random.randn(datapoints.shape[1], self.num_classes)

        batch_size = datapoints.shape[0] // num_iterations
        loss_iterations = np.zeros(num_iterations)
        for itr in np.random.permutation(range(num_iterations)) :  # random batch without replacement
            # extracting a batch from the full dataset, using the above iterator
            data_batch = datapoints[itr * batch_size : (itr + 1) * batch_size]
            labels_batch = labels[itr * batch_size : (itr + 1) * batch_size]

            scores, loss_iterations[itr] = self.forward(data_batch, labels_batch)  # calculate final scores and loss
            gradients = self.backward(data_batch, scores, labels_batch)  # calculate gradients
            """sanity check using numerical gradient calculation
            import metrics
            num_gradients = metrics.gradient_check(
                self, data_batch, self.weights, labels_batch
            )  # """
            self.weights -= learning_rate * gradients  # update weights of all layers

        return loss_iterations

    # predict an output class for given input datapoint(s)
    def predict( self, test_points ) :  # test_points can be 1d or 2d array
        axis = int(test_points.ndim == 2)
        return np.argmax(test_points @ self.weights, axis)  # N x C, if 2d input array else C-vector

    # forward pass
    def forward( self, datapoints, labels ) :
        scores = datapoints @ self.weights
        if isinstance(self.loss_model, metrics.SparseCELoss) :  # if loss is CrossEntropy, calculate softmax of scores
            scores = metrics.Softmax.forward(scores)
        loss = self.loss_model.forward(scores, labels, 0.5 * np.sum(self.weights ** 2))
        return scores, loss

    # backpropagation step
    def backward( self, datapoints, scores, labels ) :
        gradients, reg_lambda = self.loss_model.backward(scores, labels, )
        if isinstance(self.loss_model, metrics.SparseCELoss) :  # if loss is CrossEntropy, get gradients of softmax
            gradients = metrics.Softmax.backward(gradients, scores)
        gradients = (datapoints.T @ gradients) / datapoints.shape[0] + reg_lambda * self.weights
        return gradients
