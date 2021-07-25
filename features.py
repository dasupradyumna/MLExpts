import numpy as np

import metrics


# image : H x W x C numpy array
def __polar_gradient( image ) :
    gradX = np.zeros(image.shape)
    gradX[:, 1 : -1] = image[:, 2 :] - image[:, : -2]
    gradX[0, :] = gradX[-1, :] = 0

    gradY = np.zeros(image.shape)
    gradY[1 : -1, :] = image[2 :, :] - image[: -2, :]
    gradY[:, 0] = gradY[:, -1] = 0

    Mag = np.sqrt(gradX * gradX + gradY * gradY)
    Ang = np.arctan2(gradY, gradX) * 180 / np.pi
    Ang[Ang < 0] += 180
    Ang[Ang == 180] = 0

    return Mag, Ang


def __get_histograms( magnitudes, directions, hist_bins, cell_size ) :
    H, W, C = magnitudes.shape
    bins, bin_width = np.linspace(0, 180, num=hist_bins, endpoint=False, retstep=True)

    cell_grid = directions.reshape(H, W, C, 1)
    cell_grid = abs(cell_grid - bins) / bin_width
    idx = cell_grid > (90 / bin_width)
    cell_grid[idx] = hist_bins - cell_grid[idx]
    cell_grid = 1 - cell_grid
    cell_grid[cell_grid < 0] = 0
    cell_grid *= magnitudes.reshape(H, W, C, 1)

    cell_grid = cell_grid.reshape(H // cell_size, cell_size, W // cell_size, cell_size, C, hist_bins)
    return np.sum(cell_grid, axis=(1, 3))


def HOG(
    image, orientations=9, cell_size=8, norm_grid=2, *,
    color=True, norm=metrics.EuclideanNorm
) :
    if (image.shape[0] % cell_size) + (image.shape[1] % cell_size) != 0 :
        raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

    if not color : image = image.reshape(image.shape[0], image.shape[1], 1)

    mags, dirs = __polar_gradient(image)
    grid = __get_histograms(mags, dirs, orientations, cell_size)


from PIL import Image

img = Image.open("images.png").convert("L")
# img.show()
img = np.array(img.resize((192, 192))) / 255.0
HOG(img, color=False)
