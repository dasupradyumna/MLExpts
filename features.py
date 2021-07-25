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


def HOG(
    image, orientations=9, cell_size=8, norm_grid=2, *,
    color=True, norm=metrics.EuclideanNorm
) :
    H = image.shape[0]
    W = image.shape[1]
    if not color : image = image.reshape(H, W, 1)

    mags, dirs = __polar_gradient(image)

    # if H % (cell_size * norm_grid) + W % (cell_size * norm_grid) != 0 :
    #     raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

    h = H // cell_size
    w = W // cell_size
    cell_grid = dirs.reshape(h, cell_size, w, cell_size, -1, 1)
    bins, bin_width = np.linspace(0, 180, num=orientations, endpoint=False, retstep=True)
    cell_grid2 = (cell_grid - bins) / bin_width
    grid_slice = cell_grid2[abs(cell_grid2) > 90 / bin_width]
    grid_slice -= orientations * np.sign(grid_slice)
    cell_grid2[abs(cell_grid2) > 1] = 0
    pass


from PIL import Image

img = Image.open("images.png").convert("L")
# img.show()
img = np.array(img.resize((192, 192))) / 255.0
HOG(img, color=False)
