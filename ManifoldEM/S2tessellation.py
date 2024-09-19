# Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) Columbia University Hstau Liao 2018 (python version)
# Copyright (c) Columbia University Evan Seitz 2019 (python version)

from typing import Any
from nptyping import NDArray, Shape, Float64, Int
import numpy as np
from sklearn.neighbors import NearestNeighbors


def collect_nearest_neighbors(X, Q):
    """
    Finds the nearest neighbors of a set of query points Q within a dataset X and counts
    the occurrences of each point in X being the nearest neighbor.

    Parameters
    ----------
    X : np.ndarray
        An array of shape (n_samples, n_features) representing the dataset
        within which to search for nearest neighbors. Each row corresponds to a data point.
    Q : np.ndarray
        An array of shape (n_queries, n_features) representing the query points for which
        the nearest neighbors in X are to be found. Each row corresponds to a query point.

    Returns
    tuple
        np.ndarray
            An array of indices of the nearest neighbors in X for each query point in Q.
            Shape is (n_queries, 1).
        np.ndarray
            An array of counts indicating how many times each point in X is the nearest
            neighbor to the points in Q. Shape is (n_samples,).

    Notes
    -----
    - The function uses the 'ball_tree' algorithm for efficient nearest neighbor search,
      which is particularly suitable for datasets with a large number of samples and/or
      high dimensionality.
    - This could be trivially done with an all-pairs search in numba, but is not really a bottleneck
    - The `NearestNeighbors` class from scikit-learn is used to fit the model on dataset X
      and then query the nearest neighbors for points in Q.
    - The `bin_counts` array provides a histogram-like count of nearest neighbor occurrences,
      which can be useful for understanding the density or distribution of query points around
      the dataset X.
    """
    nbins = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    neighb_bins = nbrs.kneighbors(Q, return_distance=False)
    bin_counts = np.bincount(neighb_bins.squeeze(), minlength=nbins)
    return neighb_bins, bin_counts


def lovisolo_silva_tessellation(N: int) -> NDArray[Shape["*,3"], Float64]:
    """
    Distributes `N` points roughly uniformly on a unit 3-sphere using the Lovisolo-Silva algorithm.

    Parameters
    ---------
    N : int
        The number of points to attempt to distribute on the 3-sphere.

    Returns
    -------
    np.ndarray[float]
        An ndarray of shape (~N, 3) containing the coordinates of the points distributed on the 3-sphere.
        The algorithm may return slightly more or fewer points than requested due to the discretization process.

    Notes
    -----
    - The algorithm aims for a uniform distribution by adjusting the spacing between points
      based on the surface area of a unit 3-sphere and the desired number of points.
    - It iteratively adjusts the spacing (delta) to approach the target number of points,
      recalculating the distribution in each iteration until the desired number is reached
      or the maximum number of iterations (maxIter) is exceeded.
    - This method is useful for generating points for applications requiring uniform
      coverage of a spherical surface, such as sampling, simulations, or geometric analyses.

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
    Distributes `N` points roughly uniformly on a unit 3-sphere using the Fibonacci algorithm.

    Parameters
    ---------
    N : int
        The number of points to distribute on the 3-sphere.

    Returns
    -------
    np.ndarray[float]
        An ndarray of shape (N, 3) containing the coordinates of the points distributed on the 3-sphere.

    References
    ----------
    .. [1] Richard Swinbank, James Purser, "Fibonacci grids: A novel approach to global modelling,"
    Quarterly Journal of the Royal Meteorological Society, Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

    .. [2] R. Marques, C. Bouville, K. Bouatouch, and J. Blat, "Extensible Spherical Fibonacci Grids," IEEE
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
    """
    Bins unit vectors according on a sphere to a specified `bin_width` and keeps bins based on a
    threshold criterion.

    Parameters
    ----------
    S2 : np.ndarray
        An array of unit vectors to be processed. Expected shape is (3, N),
        where N is the number of points of interest (image directions). Points are assumed
        to lie above the plane such that `np.dot(plane_vec, S2[:,i]) >= 0`.
    bin_width : float
        The approximate width of each bin on the sphere, used to calculate the number of
        bins based on the surface area of the unit sphere.
    thres_low : int
        The lower threshold for the number of points in a bin to be considered significant.
        Bins with less than this parameter are discarded.
    tesselator : str
        Algorithm used to produce bins on the unit sphere. Valid options are 'lovisolo_silva' and
        'fibonacci'.
    plane_vec : ndarray, default=np.array([1.0, 0.0, 0.0])
        Vector normal to the plane where the sphere is cut. Bins such that `np.dot(plane_vec, bin_center) <= 0.0`
        are discarded.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, list[int]]
        ndarray
            An ndarray of ndarrays, where each subarray contains indices of points in S2 that fall
            into the corresponding bin.
        ndarray
            The coordinates of the centers of the bins on S2.
        ndarray
            The number of points in each bin.
        list[int]
            List of the indexes of the indexes of the bins that contain at least `thres_low` points.
    """

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
