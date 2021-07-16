import numpy as np


# both norms take flattened arrays of data of form (N, k)
# dim0 = no. of datapoints
# dim1 = data contained in one datapoint

# square-root of sum of squares of differences
def EuclideanNorm( x, y ) :
    axis = int(x.ndim == 2 or y.ndim == 2)
    return np.sqrt(np.sum((x - y) * (x - y), axis))


# sum of magnitudes of differences
def ManhattanNorm( x, y ) :
    axis = int(x.ndim == 2 or y.ndim == 2)
    return np.sum(abs(x - y), axis)


# calculates gradients numerically using first principle; sanity check for analytical gradient functions
def gradient_check( weights, datapoints, labels, loss_model ) :
    current_loss = loss_model.evaluate(weights, datapoints, labels)
    gradients = np.zeros(weights.shape)
    h = 1e-8  # perturbation value
    for row in range(weights.shape[0]) :
        for col in range(weights.shape[1]) :
            weights[row, col] += h  # disturb each weight value by h
            new_loss = loss_model.evaluate(weights, datapoints, labels)  # perturbed loss value
            gradients[row, col] = (new_loss - current_loss) / h  # first principles of differentiation
            weights[row, col] -= h  # restoring weight matrix

    return gradients


# Multi-class SVM (Hinge) Loss
# datapoints : N x K float array, labels : N x C bool array, scores : N x C float array
# methods for calculating loss and gradient
class MultiSVMLoss :

    def __init__( self, RegularizationLambda, MarginThreshold ) :
        self.reg_lambda = RegularizationLambda  # regularization hyperparameter
        self.margin = MarginThreshold  # threshold for zero loss
        self.loss_cache = None  # used in gradient calculation

    def evaluate( self, weights, datapoints, labels ) :
        scores = datapoints @ weights  # X.W matrix multiplication
        true_label_scores = scores[labels].reshape((-1, 1))  # N x 1, contains scores corresponding to true labels
        loss = np.maximum(0, scores - true_label_scores + self.margin)  # N x C, loss for each datapoint and each class
        loss[labels] = 0  # setting loss value for true labels to zero
        self.loss_cache = (loss > 0)  # caching binary loss values, loss > 0 : 1 else 0 (derivative of Hinge loss)
        loss = np.sum(loss, axis=1)  # N-vector, summing up loss over all classes for each datapoint
        loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
        loss += 0.5 * self.reg_lambda * np.sum(weights * weights)  # regularization term

        return loss

    def gradient( self, weights, datapoints, labels ) :
        if self.loss_cache is None : return np.zeros(weights.shape)

        loss_01 = self.loss_cache.astype(int)  # N x C, extracting cached loss
        # N-vector, setting loss_01 values for true labels equal to sum of ones in each row (datapoint)
        loss_01[labels] = -np.sum(loss_01, axis=1)
        gradients = (datapoints.T @ loss_01) / datapoints.shape[0]  # multiplying by X, dividing by no. of datapoints
        gradients += self.reg_lambda * weights  # derivative of regularization term

        return gradients


# Cross-entropy Loss, using softmax
# datapoints : N x K float array, labels : N x C bool array, scores : N x C float array
# methods for calculating loss and gradient
class CrossEntropyLoss :

    def __init__( self, RegularizationLambda ) :
        self.reg_lambda = RegularizationLambda  # regularization hyperparameter
        self.softmax_cache = None  # used in gradient calculation

    def evaluate( self, weights, datapoints, labels ) :
        softmax_scores = np.exp(datapoints @ weights)  # N x C, e^s for each element s from X.W matrix multiplication
        softmax_scores = softmax_scores / np.sum(softmax_scores, axis=1, keepdims=True)  # N x C, softmax distributions
        self.softmax_cache = softmax_scores  # caching softmax scores
        loss = -np.log(softmax_scores[labels])  # N-vector, logs of scores of true labels
        loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
        loss += 0.5 * self.reg_lambda * np.sum(weights * weights)  # regularization term

        return loss

    def gradient( self, weights, datapoints, labels ) :
        if self.softmax_cache is None : return np.zeros(weights.shape)

        softmax_scores = self.softmax_cache  # extracting cached scores
        softmax_scores[labels] -= 1  # subtracting only scores corresponding to true class for each datapoint by 1
        gradients = (datapoints.T @ softmax_scores) / datapoints.shape[0]  # derivative of CE Loss
        gradients += self.reg_lambda * weights  # derivative of regularization term
        self.softmax_cache = None

        return gradients
