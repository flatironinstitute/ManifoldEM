# Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) Columbia University Hstau Liao 2018 (python version)
# Copyright (c) Columbia University Evan Seitz 2019 (python version)

from typing import Any
from nptyping import NDArray, Shape, Float64, Int
import numpy as np
from sklearn.neighbors import NearestNeighbors


def collect_nearest_neighbors(X, Q):
    nbins = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    neighb_bins = nbrs.kneighbors(Q, return_distance=False)
    bin_counts = np.bincount(neighb_bins.squeeze(), minlength=nbins)
    return neighb_bins, bin_counts


def lovisolo_silva_tessellation(N: int) -> NDArray[Shape["*,3"], Float64]:
    """
    distributes numPts points roughly uniformly on a unit 3-sphere and
    returns the coordinates in results. Number of iterations required is
    returned in iter.

    Algorithm adapted from L. Lovisolo and E.A.B. da Silva, Uniform
    distribution of points on a hyper-sphere with applications to vector
    bit-plane encoding, IEE Proc.-Vis. Image Signal Process., Vol. 148, No.
    3, June 2001

    Programmed February 2009
    Copyright (c) Russell Fung 2009
    Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """
    maxIter = 100
    K = N
    A3 = 4 * np.pi  # surface area of a unit 3-sphere
    delta = np.sqrt(A3 / K)
    results = np.zeros((2 * K, 3))
    # algorithm sometimes returns more/ less points
    it = 0
    id = 0

    while id != K and it < maxIter:
        it = it + 1
        id = 0
        dw1 = delta
        for w1 in np.arange(0.5 * dw1, np.pi, dw1):
            cw1 = np.cos(w1)
            sw1 = np.sin(w1)
            x1 = cw1
            dw2 = dw1 / sw1
            for w2 in np.arange(0.5 * dw2, 2 * np.pi, dw2):
                cw2 = np.cos(w2)
                sw2 = np.sin(w2)
                x2 = sw1 * cw2
                x3 = sw1 * sw2

                results[id, :] = np.hstack((x1, x2, x3))
                id = id + 1

        delta = delta * np.sqrt(id / K)

    results = results[0:K, :]
    return results


def fibonacci_tessellation(N: int) -> NDArray[Shape["*,3"], Float64]:
    """
X = get_fibonacci(N)

Returns `X` a (N,3) array of coordinates of `N` Fibonacci points on S^2
(the unit sphere). `N` may be any positive integer.

See:
Richard Swinbank, James Purser, "Fibonacci grids: A novel approach to global modelling,"
Quarterly Journal of the Royal Meteorological Society, Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

R. Marques, C. Bouville, K. Bouatouch, and J. Blat, "Extensible Spherical Fibonacci Grids," IEEE
Transactions on Visualization and Computer Graphics, vol. 27, no. 4, pp. 2341–2354, 2021 doi:
10.1109/TVCG.2019.2952131
"""
    phi = (1 + np.sqrt(5.0)) / 2
    dphi = 2 * np.pi / phi  # azimuth regular spacing
    X = np.zeros(shape=(N, 3))
    for j in range(N):  # loop over pts
        X[j, 2] = 1 - (2 * j + 1) / N  # z, symmetrically shifted
        phi = dphi * j  # offset irrelevant
        s, c = np.sin(phi), np.cos(phi)
        rho = np.sqrt(1.0 - X[j, 2]**2)
        X[j, 0] = rho * c
        X[j, 1] = rho * s

    return X


def bin_and_threshold(
    S2: NDArray[Shape["3,*"], Float64],
    bin_width: float,
    thres_low: int,
    tessellator: str,
    plane_vec: NDArray[Shape["3"], Float64] = np.array([1.0, 0.0, 0.0])
) -> tuple[NDArray[Shape["*"], Any], NDArray[Shape["3,*"], Float64], NDArray[Shape["*"], Int], list[int]]:
    # Attempt to bin sphere in equal patches with area ~bin_width^2
    requested_n_bins = int(4 * np.pi / (bin_width**2))
    if tessellator == "lovisolo_silva":
        bin_centers = lovisolo_silva_tessellation(requested_n_bins).T
    elif tessellator == "fibonacci":
        bin_centers = fibonacci_tessellation(requested_n_bins).T
    else:
        errorstr = f'Invalid tesselator supplied ({tessellator}). Valid options are ["lovisolo_silva", "fibonacci"]'
        raise ValueError(errorstr)

    n_bins = bin_centers.shape[1]

    mask = np.zeros(shape=(n_bins), dtype=bool)
    for i in range(n_bins):
        if np.dot(bin_centers[:, i], plane_vec) >= 0.0:
            mask[i] = True

    bin_centers = bin_centers[:, mask]
    n_bins = bin_centers.shape[1]

    # For each point in S2, find closest bin
    neighb_bins, n_points_in_bin = collect_nearest_neighbors(bin_centers.T, S2.T)

    # For each bin, list all points in S2 that live in that bin
    neighb_list = [[] for _ in range(n_bins)]
    for i, index in enumerate(neighb_bins.ravel()):
        neighb_list[index].append(i)
    neighb_list = np.array([np.array(a) for a in neighb_list], dtype=object)

    # list bins that have more than thres_low points
    prd_indices = []
    for pd_rel, n_points in enumerate(n_points_in_bin):
        if n_points >= thres_low:
            prd_indices.append(pd_rel)

    return (neighb_list, bin_centers, n_points_in_bin, prd_indices)
