import numpy as np

# both norms take flattened arrays of data of form (N, k)
# dim0 = no. of datapoints
# dim1 = data contained in one datapoint

# square-root of sum of squares of differences
def EuclideanNorm( x, y ) :
    k = 1 if x.ndim > 1 or y.ndim > 1 else 0
    return np.sqrt(np.sum((x-y)**2, axis=k))

# sum of magnitudes of differences
def ManhattanNorm( x, y ) :
    k = 1 if x.ndim > 1 or y.ndim > 1 else 0
    return np.sum(abs(x-y), axis=k)

def MultiSVMLoss():
    pass

def SoftmaxLoss():
    pass
