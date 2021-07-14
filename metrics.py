import numpy as np


# both norms take flattened arrays of data of form (N, k)
# dim0 = no. of datapoints
# dim1 = data contained in one datapoint

# square-root of sum of squares of differences
def EuclideanNorm( x, y ) :
    k = 1 if x.ndim > 1 or y.ndim > 1 else 0
    return np.sqrt(np.sum((x - y) ** 2, axis=k))


# sum of magnitudes of differences
def ManhattanNorm( x, y ) :
    k = 1 if x.ndim > 1 or y.ndim > 1 else 0
    return np.sum(abs(x - y), axis=k)


def MultiSVMLossWithGrad( datapoints, labels, scores ) :
    # scores : N x C float array, labels : N x C bool array
    true_label_scores = scores[labels].reshape((-1, 1))  # N x 1, contains scores corresponding to true labels
    loss = np.maximum(0, scores - true_label_scores + 1)  # N x C, loss for each datapoint and each class label
    loss[labels] = 0  # setting loss value for true labels to zero
    loss_01 = (loss > 0).astype(int)  # N x C, loss > 0 : 1 else 0 (derivative of Hinge loss)
    loss = np.sum(loss, axis=1)  # N-vector, summing up loss over all classes for each datapoint
    loss = np.mean(loss)  # scalar, taking average of losses over all datapoints

    # N-vector, setting loss_01 values for true labels equal to sum of ones in each row (datapoint)
    loss_01[labels] = -np.sum(loss_01, axis=1)
    gradients = datapoints.T @ loss_01

    return loss, gradients


def SoftmaxLoss( ) :
    pass
