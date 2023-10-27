import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ManifoldEM.core import distribute3Sphere
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def quaternion_to_S2(q):
    try:
        assert (q.shape[0] > 3)
    except AssertionError:
        _logger.error('subroutine get_S2: q has wrong dimensions')
        _logger.exception('subroutine get_S2: q has wrong dimensions')
        raise

    # projection angles
    S2 = 2*np.vstack((q[1, :]*q[3, :] - q[0, :]*q[2, :],
                      q[0, :]*q[1, :] + q[2, :]*q[3, :],
                      q[0, :]**2 + q[3, :]**2 - 0.5))
    return S2


def collect_nearest_neighbors(X, Q):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    _, neighb_bins = nbrs.kneighbors(Q)
    bin_counts = np.bincount(neighb_bins.squeeze())
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

    # list bins that have more than thres_low points AND lie on bigger half of "mid"
    conjugate_bins = []
    mid = n_bins // 2
    take_lower = not bool(n_bins % 2)
    if take_lower:
        pd = 0  # PD index
        for i in n_points_in_bin[:mid]:
            if i >= thres_low:
                conjugate_bins.append(pd)
            pd += 1
    else:
        pd = mid  # PD index
        for i in n_points_in_bin[mid:]:
            if i >= thres_low:
                conjugate_bins.append(pd)
            pd += 1

    return (neighb_list, S2, bin_centers, n_points_in_bin, conjugate_bins)
