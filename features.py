import numpy as np


# calculates the X and Y gradients using simple [-1 0 1] kernels
# transforms into polar gradients, given by magnitude and angle
def __polar_gradient( image ) :
    gradX = np.zeros_like(image)  # cartesian X gradient (along columns)
    gradX[:, :, 1 : -1] = image[:, :, 2 :] - image[:, :, : -2]
    gradX[:, 0, :] = gradX[:, -1, :] = 0  # edge columns are all zero

    gradY = np.zeros_like(image)  # cartesian Y gradient (along rows)
    gradY[:, 1 : -1, :] = image[:, 2 :, :] - image[:, : -2, :]
    gradY[:, :, 0] = gradY[:, :, -1] = 0  # edge rows are all zero

    Mag = np.hypot(gradX, gradY)  # magnitude of X and Y gradients
    Ang = np.rad2deg(np.arctan2(gradY, gradX)) % 180  # angles of X and Y gradients (between 0 and 180 degrees)

    # for every pixel, choosing only the channel with the highest gradient magnitude for final gradient matrix
    max_grad_mask = Mag.argmax(axis=-1)[..., np.newaxis] == np.arange(image.shape[-1])
    Mag = Mag[max_grad_mask].reshape(*image.shape[:-1])
    Ang = Ang[max_grad_mask].reshape(*image.shape[:-1])
    return Mag, Ang


# uses input polar gradients to populate a histogram
# bins of histogram are given by the angles and the value populated in each bin is the corresponding gradient value
def __get_histograms( magnitudes, directions, hist_bins, cell_size ) :
    bins, bin_width = np.linspace(0, 180, num=hist_bins, endpoint=False, retstep=True)  # bin values for angles

    cell_grid = directions[..., np.newaxis]  # angles of the gradients
    cell_grid = abs(cell_grid - bins) / bin_width  # fractional distance of each angle from the bins
    idx = cell_grid > (90 / bin_width)  # find all distances more than half the number of bins
    cell_grid[idx] = hist_bins - cell_grid[idx]  # making the distances symmetric (triangular distribution)
    # for each angle, makes all bins negative except the (1 or) 2 bins closest to the angle
    # the 2 positive values add up to 1, bilinear interpolation ratio
    cell_grid = 1 - cell_grid
    cell_grid[cell_grid < 0] = 0  # get rid of the negative bin values
    cell_grid *= magnitudes[..., np.newaxis]  # add the weights to each bin vector (gradient magnitudes)

    # accumulate all the bin vectors of a cell into a single vector
    N, H, W = magnitudes.shape
    cell_grid.resize(N, H // cell_size, cell_size, W // cell_size, cell_size, hist_bins)
    return np.sum(cell_grid, axis=(2, 4))


# normalizes the histograms by using blocks of cells
# each cell's normalized vector is added to final feature set
def __normalize_histograms( grid, norm_block ) :
    # generating vectorized indices for sliding window
    N, H, W, bins = grid.shape
    idxH = np.arange(H - norm_block + 1)[:, np.newaxis] + np.arange(norm_block)
    idxH = idxH[:, np.newaxis, :, np.newaxis]  # index the height dimension
    idxW = np.arange(W - norm_block + 1)[:, np.newaxis] + np.arange(norm_block)
    idxW = idxW[np.newaxis, :, np.newaxis, :]  # index the width dimension
    idxN = np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]  # index of datapoints dimension

    blocks = grid[idxN, idxH, idxW]
    blocks.resize(*blocks.shape[:3], norm_block * norm_block * bins)  # merging all the cell vectors in a block
    blocks_norm = np.linalg.norm(blocks, axis=-1, keepdims=True)  # L2 norm of each block's bin vectors
    np.divide(blocks, blocks_norm, out=blocks, where=(blocks_norm != 0))  # dividing with norm, only for nonzero values
    return blocks.reshape(N, -1).squeeze()  # remove singleton dimensions


# Histogram of Oriented Gradients
# extracts features from a sequence of images (float numpy arrays)
def HOG( images, **kwargs ) :
    orientations = kwargs.pop("orientations", 9)  # number of bins for angles to split into
    cell_size = kwargs.pop("cell_size", 8)  # number of pixels per cell for calculating histogram
    norm_block = kwargs.pop("norm_block", 2)  # number of cells per block for normalization
    color = kwargs.pop("color", True)  # grayscale or color image
    gamma_correction = kwargs.pop("gamma_correction", False)  # apply gamma correction on input image
    if len(kwargs) != 0 :
        raise ValueError("Too many arguments to unpack from function call.\n")

    if not color : images = images[..., np.newaxis]  # add singleton dimension if gray scale image
    if images.ndim == 3 : images = images[np.newaxis, ...]  # add singleton dimension if single image is input

    # throw an error if the image can not be split into a whole number of cells
    if (images.shape[1] % cell_size) + (images.shape[2] % cell_size) != 0 :
        raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

    if gamma_correction : images = np.sqrt(images)  # gamma correction
    mags, dirs = __polar_gradient(images)  # calculate magnitude and orientations of gradient
    grid = __get_histograms(mags, dirs, orientations, cell_size)  # get initial histogram
    return __normalize_histograms(grid, norm_block)  # normalize above output histogram


if __name__ == "__main__" :
    from PIL import Image
    from skimage.feature import hog

    img = Image.open("images.png").convert("RGB")
    img.show()
    img = np.array(img.resize((64, 128))) / 255.0
    custom = HOG(img)
    inbuilt = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
    print(np.linalg.norm(inbuilt - custom, ord=1) / inbuilt.size)
