# Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) Columbia University Hstau Liao 2018 (python version)
# Copyright (c) Columbia University Evan Seitz 2019 (python version)

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ManifoldEM.core import distribute3Sphere

def quaternion_to_S2(q):
    # TODO: Understand how this magically gets rid of final psi rotation
    S2 = 2*np.vstack((q[1, :]*q[3, :] - q[0, :]*q[2, :],
                      q[0, :]*q[1, :] + q[2, :]*q[3, :],
                      q[0, :]**2 + q[3, :]**2 - 0.5))
    return S2


def collect_nearest_neighbors(X, Q):
    nbins = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    neighb_bins = nbrs.kneighbors(Q, return_distance=False)
    bin_counts = np.bincount(neighb_bins.squeeze(), minlength=nbins)
    return neighb_bins, bin_counts


def bin_and_threshold(q, bin_width, thres_low, thres_high):
    # Attempt to bin sphere in equal patches with area ~bin_width^2
    requested_n_bins = int(4 * np.pi / (bin_width**2))
    bin_centers = distribute3Sphere(requested_n_bins)[0].T
    n_bins = bin_centers.shape[1]

    # Map quaternions onto unit_vectors (S2)
    S2 = quaternion_to_S2(q)

    # For each point in S2, find closest bin
    neighb_bins, n_points_in_bin = collect_nearest_neighbors(bin_centers.T, S2.T)

    # For each bin, list all points in S2 that live in that bin
    neighb_list = [[] for _ in range(n_bins)]
    for i, index in enumerate(neighb_bins.ravel()):
        neighb_list[index].append(i)
    neighb_list = np.array([np.array(a) for a in neighb_list], dtype=object)

    # list bins that have more than thres_low points AND lie on bigger half of bin list
    conjugate_bins = []
    start_bin, end_bin = (n_bins // 2, n_bins) if n_bins % 2 else (0, n_bins // 2)
    for pd_rel, n_points in enumerate(n_points_in_bin[start_bin:end_bin]):
        if n_points >= thres_low:
            conjugate_bins.append(pd_rel + start_bin)

    return (neighb_list, S2, bin_centers, n_points_in_bin, conjugate_bins)
