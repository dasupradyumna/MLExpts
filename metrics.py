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
def gradient_check( weights, datapoints, labels, margin, reg_lambda, loss ) :
    current_loss = loss(datapoints, labels, weights, margin, reg_lambda, calc_grad=False)
    gradients = np.zeros(weights.shape)
    h = 1e-8  # perturbation value
    for row in range(weights.shape[0]) :
        for col in range(weights.shape[1]) :
            weights[row, col] += h  # disturb each weight value by h
            new_loss = loss(datapoints, labels, weights, margin, reg_lambda, calc_grad=False)  # perturbed loss value
            gradients[row, col] = (new_loss - current_loss) / h  # first principles of differentiation
            weights[row, col] -= h  # restoring weight matrix

    return gradients


# Multi-class SVM (Hinge) Loss function, with option to calculate gradient
# datapoints : N x K float array, labels : N x C bool array, scores : N x C float array
# margin : threshold for accepting a particular score as satisfactorily true
# returns the calculated loss, and optionally also the gradient for current weights
def MultiSVMLossWithGrad( datapoints, labels, weights, margin, reg_lambda, calc_grad=True ) :
    gradients = None
    scores = datapoints @ weights  # X.W matrix multiplication
    true_label_scores = scores[labels].reshape((-1, 1))  # N x 1, contains scores corresponding to true labels
    loss = np.maximum(0, scores - true_label_scores + margin)  # N x C, loss for each datapoint and each class label
    loss[labels] = 0  # setting loss value for true labels to zero

    if calc_grad :
        loss_01 = (loss > 0).astype(int)  # N x C, loss > 0 : 1 else 0 (derivative of Hinge loss)
        # N-vector, setting loss_01 values for true labels equal to sum of ones in each row (datapoint)
        loss_01[labels] = -np.sum(loss_01, axis=1)
        gradients = (datapoints.T @ loss_01) / datapoints.shape[0]  # multiplying by X, dividing by no. of datapoints
        gradients += reg_lambda * weights  # derivative of regularization term

    loss = np.sum(loss, axis=1)  # N-vector, summing up loss over all classes for each datapoint
    loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
    loss += 0.5 * reg_lambda * np.sum(weights * weights)  # regularization term

    return (loss, gradients) if calc_grad else loss


def CrossEntropyLossWithGrad( datapoints, labels, weights, dummy_margin, reg_lambda, calc_grad=True ) :
    gradients = None
    softmax_scores = np.exp(datapoints @ weights)  # N x C, e^s for each element s from X.W matrix multiplication
    softmax_scores = softmax_scores / np.sum(softmax_scores, axis=1, keepdims=True)  # N x C, softmax distributions
    loss = -np.log(softmax_scores[labels])  # N-vector, logs of scores of true labels
    loss = np.mean(loss)  # scalar, taking average of losses over all datapoints
    loss += 0.5 * reg_lambda * np.sum(weights * weights)  # regularization term

    if calc_grad :
        softmax_scores[labels] -= 1  # subtracting only scores corresponding to true class for each datapoint by 1
        gradients = (datapoints.T @ softmax_scores) / datapoints.shape[0]  # derivative of CE Loss
        gradients += reg_lambda * weights  # derivative of regularization term

    return (loss, gradients) if calc_grad else loss
