import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter

# original version adapted from "scivision pyoptflow",
# modified and customized by Suvrajit Maji

HSKERN = np.array([[1.0 / 12, 1.0 / 6, 1.0 / 12], [1.0 / 6, 0., 1.0 / 6], [1.0 / 12, 1.0 / 6, 1.0 / 12]], float)
kernelX = np.array([[-1.0, 1.0], [-1.0, 1.0]]) * .25  # kernel for computing d/dx
kernelY = np.array([[-1.0, -1.0], [1.0, 1.0]]) * .25  # kernel for computing d/dy
kernelT = np.ones((2, 2)) * .25

def filter2(image, kernel):
    return cv.filter2D(image, -1, kernel, borderType=cv.BORDER_REFLECT)


def computeDerivatives(im1, im2):
    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft


def lowpassfilt(im, sig):
    im = gaussian_filter(im, sigma=sig)
    return im


def op(im1, im2, uInitial, vInitial, sigma=1.0, alpha=0.001, Niter=8, verbose=False):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iterations
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    im1 = lowpassfilt(im1, sigma)
    im2 = lowpassfilt(im2, sigma)

    # set up initial velocities
    if uInitial.shape[0] < 2:
        uInitial = np.zeros_like(im1)
    if vInitial.shape[0] < 2:
        vInitial = np.zeros_like(im1)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Iteration to reduce error
    for it in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
        # common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        # iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V
