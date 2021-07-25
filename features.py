import numpy as np

import metrics


# image : H x W (x C) numpy array
def __X_Y_gradient( image ) :
    gradX = np.zeros(image.shape)
    gradX[:, 1 : -1] = image[:, 2 :] - image[:, : -2]
    gradX[0, :] = gradX[-1, :] = 0

    gradY = np.zeros(image.shape)
    gradY[1 : -1, :] = image[: -2, :] - image[2 :, :]
    gradY[:, 0] = gradY[:, -1] = 0

    return gradX, gradY


def HOG(
    image, orientations=9, cell_size=8, norm_grid=2, *,
    color=True, norm=metrics.EuclideanNorm
) :
    gradX, gradY = __X_Y_gradient(image)
    mags = np.sqrt(gradX * gradX + gradY * gradY)  # H x W (x C) array
    dirs = np.atan(gradY / gradX)

    if (image.shape[0] % cell_size) + (image.shape[1] % cell_size) != 0 :
        raise ValueError("Image cannot be divided into exact cells. Check dimensions.\n")

    if color :
        cellH = image.shape[0] // cell_size
        cellW = image.shape[1] // cell_size
        C = image.shape[2]
        hog = np.zeros((cellH, cellW, C, orientations))
        for r in range(cellH) :
            for c in range(cellW) :
                for d in range(C) :
                    for i in range(cell_size) :
                        pass
