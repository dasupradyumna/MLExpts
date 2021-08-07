import numpy as np


# Histogram of Oriented Gradients
class HOG :

    def __init__( self, **kwargs ) :
        self.orientations = kwargs.pop("orientations", 9)  # number of bins for angles to split into
        self.cell_size = kwargs.pop("cell_size", 8)  # number of pixels per cell for calculating histogram
        self.norm_block = kwargs.pop("norm_block", 2)  # number of cells per block for normalization
        self.color = kwargs.pop("color", True)  # grayscale or color image
        self.gamma_correction = kwargs.pop("gamma_correction", False)  # apply gamma correction on input image
        if len(kwargs) != 0 :
            raise ValueError("Too many arguments to unpack from function call.\n")

    # extracts features from a sequence of images (float numpy arrays)
    def __call__( self, images ) :
        if not self.color : images = images[..., np.newaxis]  # add singleton dimension if gray scale image
        if images.ndim == 3 : images = images[np.newaxis, ...]  # add singleton dimension if single image is input

        # throw an error if the image can not be split into a whole number of cells
        if (images.shape[1] % self.cell_size) + (images.shape[2] % self.cell_size) != 0 :
            raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

        if self.gamma_correction : images = np.sqrt(images)  # gamma correction
        mags, dirs = self.__polar_gradient(images)  # calculate magnitude and orientations of gradient
        grid = self.__get_histograms(mags, dirs)  # get initial histogram
        return self.__normalize_histograms(grid)  # normalize above output histogram

    # calculates the X and Y gradients using simple [-1 0 1] kernels
    # transforms into polar gradients, given by magnitude and angle
    @staticmethod
    def __polar_gradient( images ) :
        gradX = np.zeros_like(images)  # cartesian X gradient (along columns)
        gradX[:, :, 1 : -1] = images[:, :, 2 :] - images[:, :, : -2]
        gradX[:, 0, :] = gradX[:, -1, :] = 0  # edge columns are all zero

        gradY = np.zeros_like(images)  # cartesian Y gradient (along rows)
        gradY[:, 1 : -1, :] = images[:, 2 :, :] - images[:, : -2, :]
        gradY[:, :, 0] = gradY[:, :, -1] = 0  # edge rows are all zero

        Mag = np.hypot(gradX, gradY)  # magnitude of X and Y gradients
        Ang = np.rad2deg(np.arctan2(gradY, gradX)) % 180  # angles of X and Y gradients (between 0 and 180 degrees)

        # for every pixel, choosing only the channel with the highest gradient magnitude for final gradient matrix
        max_grad_mask = Mag.argmax(axis=-1)[..., np.newaxis] == np.arange(images.shape[-1])
        Mag = Mag[max_grad_mask].reshape(*images.shape[:-1])
        Ang = Ang[max_grad_mask].reshape(*images.shape[:-1])
        return Mag, Ang

    # uses input polar gradients to populate a histogram
    # bins of histogram are given by the angles and the value populated in each bin is the corresponding gradient value
    def __get_histograms( self, magnitudes, directions ) :
        bins, bin_width = np.linspace(0, 180, num=self.orientations, endpoint=False, retstep=True)  # bins for angles

        cell_grid = directions[..., np.newaxis]  # angles of the gradients
        cell_grid = abs(cell_grid - bins) / bin_width  # fractional distance of each angle from the bins
        idx = cell_grid > (90 / bin_width)  # find all distances more than half the number of bins
        cell_grid[idx] = self.orientations - cell_grid[idx]  # making the distances symmetric (triangular distribution)
        # for each angle, makes all bins negative except the (1 or) 2 bins closest to the angle
        # the 2 positive values add up to 1, bilinear interpolation ratio
        cell_grid = 1 - cell_grid
        cell_grid[cell_grid < 0] = 0  # get rid of the negative bin values
        cell_grid *= magnitudes[..., np.newaxis]  # add the weights to each bin vector (gradient magnitudes)

        # accumulate all the bin vectors of a cell into a single vector
        N, H, W = magnitudes.shape
        cell_grid.resize(N, H // self.cell_size, self.cell_size, W // self.cell_size, self.cell_size, self.orientations)
        return np.sum(cell_grid, axis=(2, 4))

    # normalizes the histograms by using blocks of cells
    # each cell's normalized vector is added to final feature set
    def __normalize_histograms( self, grid ) :
        # generating vectorized indices for sliding window
        N, H, W, bins = grid.shape
        idxH = np.arange(H - self.norm_block + 1)[:, np.newaxis] + np.arange(self.norm_block)
        idxH = idxH[:, np.newaxis, :, np.newaxis]  # index the height dimension
        idxW = np.arange(W - self.norm_block + 1)[:, np.newaxis] + np.arange(self.norm_block)
        idxW = idxW[np.newaxis, :, np.newaxis, :]  # index the width dimension
        idxN = np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]  # index of datapoints dimension

        blocks = grid[idxN, idxH, idxW]
        blocks.resize(*blocks.shape[:3], self.norm_block * self.norm_block * bins)  # merging all the cell in a block
        blocks_norm = np.linalg.norm(blocks, axis=-1, keepdims=True)  # L2 norm of each block's bin vectors
        blocks = blocks / (blocks_norm + 1e-8)  # dividing by norm
        return blocks.reshape(N, -1).squeeze()  # remove singleton dimensions


if __name__ == "__main__" :
    from PIL import Image
    from skimage.feature import hog

    img = Image.open("images.png").convert("RGB")
    img.show()
    img = np.array(img.resize((64, 128))) / 255.0
    hog_fd = HOG()
    custom = hog_fd(img)
    inbuilt = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
    print(np.linalg.norm(inbuilt - custom, ord=1) / inbuilt.size)
