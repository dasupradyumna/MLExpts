import numpy as np


# both norms take flattened arrays of data of form (N, k)
# dim0 = no. of datapoints
# dim1 = data contained in one datapoint

# square-root of sum of squares of differences
def EuclideanNorm( x, y ) :
    assert x.ndim <= 2 and y.ndim <= 2, \
        "Expected argument is at most a 2D numpy array."
    axis = int(x.ndim == 2 or y.ndim == 2)
    return np.sqrt(np.sum((x - y) * (x - y), axis))


# sum of magnitudes of differences
def ManhattanNorm( x, y ) :
    assert x.ndim <= 2 and y.ndim <= 2, \
        "Expected argument is at most a 2D numpy array."
    axis = int(x.ndim == 2 or y.ndim == 2)
    return np.sum(abs(x - y), axis)


# calculates gradients numerically using first principle; sanity check for analytical gradient functions
# datapoints : input to model, weights : target of gradient, labels : output of model
def gradient_check( model, datapoints, weights, labels ) :
    _, current_loss = model.forward(datapoints, labels)
    gradients = np.zeros_like(weights)
    h = 1e-8  # perturbation value
    for row in range(gradients.shape[0]) :
        for col in range(gradients.shape[1]) :
            weights[row, col] += h  # disturb each weight value by h
            _, new_loss = model.forward(datapoints, labels)  # perturbed loss value
            gradients[row, col] = (new_loss - current_loss) / h  # first principles of differentiation
            weights[row, col] -= h  # restoring weight matrix

    return gradients


# Multi-class SVM (Hinge) Loss
class MultiSVMLoss :

    def __init__( self, RegularizationLambda, MarginThreshold ) :
        self.reg_lambda = RegularizationLambda  # regularization hyperparameter
        self.margin = MarginThreshold  # threshold for zero loss
        self.loss_cache = None  # used in gradient calculation

    # calculate loss based from the output scores from model
    # scores : N x C array, labels : N-vector, reg_loss : loss summed up from regularization of weights
    def forward( self, scores, labels, reg_loss ) :
        # N x 1, contains scores corresponding to true labels
        true_label_scores = scores[np.arange(labels.size), labels][:, np.newaxis]
        loss = np.maximum(0, scores - true_label_scores + self.margin)  # N x C, loss for each datapoint and each class
        loss[np.arange(labels.size), labels] = 0  # setting loss value for true labels to zero
        self.loss_cache = (loss > 0)  # caching binary loss values, loss > 0 : 1 else 0 (derivative of Hinge loss)
        loss = np.sum(loss, axis=1)  # N-vector, summing up loss over all classes for each datapoint
        loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
        loss += self.reg_lambda * reg_loss  # regularization term

        return loss

    # calculate gradients of loss wrt the output scores of model
    # labels : N-vector
    def backward( self, _, labels ) :
        gradients = self.loss_cache.astype(int)  # N x C, extracting cached loss
        # N-vector, setting loss_01 values for true labels equal to sum of ones in each row (datapoint)
        gradients[np.arange(labels.size), labels] = -np.sum(gradients, axis=1)

        return gradients, self.reg_lambda


# Cross-Entropy Loss, implemented without one-hot vectors (sparse)
class SparseCELoss :

    def __init__( self, RegularizationLambda ) :
        self.reg_lambda = RegularizationLambda  # regularization hyperparameter

    # calculate loss based from the output scores from model
    # softmax_scores : N x C array, labels : N-vector, reg_loss : loss summed up from regularization of weights
    def forward( self, softmax_scores, labels, reg_loss ) :
        loss = - np.log(softmax_scores[np.arange(labels.size), labels])
        loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
        loss += self.reg_lambda * reg_loss  # regularization term

        return loss

    # calculate gradients of loss wrt the output scores of model
    # softmax_scores : N x C array, labels : N-vector
    def backward( self, softmax_scores, labels ) :
        gradients = np.zeros_like(softmax_scores)
        gradients[np.arange(labels.size), labels] = - 1 / softmax_scores[np.arange(labels.size), labels]
        return gradients, self.reg_lambda


# Softmax activation
class Softmax :

    # calculates softmax distribution from scores
    # scores : N x C array
    @staticmethod
    def forward( scores ) :
        softmax_scores = np.exp(scores)  # N x C, e^s for each element s from X.W matrix multiplication
        return softmax_scores / np.sum(softmax_scores, axis=-1, keepdims=True)  # N x C, softmax distributions

    # calculates gradients of distribution wrt scores
    @staticmethod
    def backward( gradients, scores ) :
        num_nodes = scores.shape[1]
        softmax_grads = - scores[:, :, np.newaxis] @ scores[:, np.newaxis, :]
        idx = np.arange(num_nodes)
        softmax_grads[:, idx, idx] += scores[:, idx]
        gradients = gradients[:, np.newaxis, :] @ softmax_grads
        return gradients.squeeze()


# ReLU activation
class ReLU :

    # zeroes out all negative values in scores : N x C array
    @staticmethod
    def forward( scores ) :
        return np.maximum(0, scores)

    # zeroes out gradients of all scores which are zero
    @staticmethod
    def backward( gradients, scores ) :
        gradients[scores == 0] = 0
        return gradients
