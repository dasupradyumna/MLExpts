import numpy as np


def __polar_gradient( image ) :
    gradX = np.zeros_like(image)
    gradX[:, :, 1 : -1] = image[:, :, 2 :] - image[:, :, : -2]
    gradX[:, 0, :] = gradX[:, -1, :] = 0

    gradY = np.zeros_like(image)
    gradY[:, 1 : -1, :] = image[:, 2 :, :] - image[:, : -2, :]
    gradY[:, :, 0] = gradY[:, :, -1] = 0

    Mag = np.sqrt(gradX * gradX + gradY * gradY)
    Ang = np.arctan2(gradY, gradX) * 180 / np.pi
    Ang[Ang < 0] += 180
    Ang[Ang == 180] = 0

    max_grad_mask = Mag.argmax(axis=-1)[..., np.newaxis] == np.arange(image.shape[-1])
    Mag = Mag[max_grad_mask].reshape(*image.shape[:-1])
    Ang = Ang[max_grad_mask].reshape(*image.shape[:-1])
    return Mag, Ang


def __get_histograms( magnitudes, directions, hist_bins, cell_size ) :
    bins, bin_width = np.linspace(0, 180, num=hist_bins, endpoint=False, retstep=True)

    cell_grid = directions[..., np.newaxis]
    cell_grid = abs(cell_grid - bins) / bin_width
    idx = cell_grid > (90 / bin_width)
    cell_grid[idx] = hist_bins - cell_grid[idx]
    cell_grid = 1 - cell_grid
    cell_grid[cell_grid < 0] = 0
    cell_grid *= magnitudes[..., np.newaxis]

    N, H, W = magnitudes.shape
    cell_grid.resize(N, H // cell_size, cell_size, W // cell_size, cell_size, hist_bins)
    return np.sum(cell_grid, axis=(2, 4))


def __normalize_histograms( grid, norm_block ) :
    N, H, W, bins = grid.shape
    idxH = np.arange(H - norm_block + 1)[:, np.newaxis] + np.arange(norm_block)
    idxH = idxH[:, np.newaxis, :, np.newaxis]
    idxW = np.arange(W - norm_block + 1)[:, np.newaxis] + np.arange(norm_block)
    idxW = idxW[np.newaxis, :, np.newaxis, :]
    idxN = np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    blocks = grid[idxN, idxH, idxW]
    blocks.resize(*blocks.shape[:3], norm_block * norm_block, bins)
    blocks_norm = np.linalg.norm(blocks, axis=(1, 2), keepdims=True)
    np.divide(blocks, blocks_norm, out=blocks, where=(blocks_norm != 0))
    return blocks.reshape(N, -1)


def HOG( image, orientations=9, cell_size=8, *, norm_block=2, color=True ) :
    if not color : image = image[..., np.newaxis]
    if image.ndim == 3 : image = image[np.newaxis, ...]

    if (image.shape[1] % cell_size) + (image.shape[2] % cell_size) != 0 :
        raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

    mags, dirs = __polar_gradient(image)
    grid = __get_histograms(mags, dirs, orientations, cell_size)
    return __normalize_histograms(grid, norm_block)


if __name__ == "__main__" :
    from PIL import Image
    from skimage.feature import hog

    img = Image.open("images.png").convert("RGB")
    img.show()
    img = np.array(img.resize((64, 128))) / 255.0
    custom = HOG(img)
    inbuilt = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
    print(np.linalg.norm(inbuilt - custom, ord=1) / inbuilt.size)
