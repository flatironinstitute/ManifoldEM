import logging
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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


def _get_S2(q):
    try:
        assert (q.shape[0] > 3)
    except AssertionError:
        _logger.error('subroutine get_S2: q has wrong dimensions')
        _logger.exception('subroutine get_S2: q has wrong diemnsions')
        sys.exit(1)
        raise
        # projection angles
    S2 = 2*np.vstack((q[1, :]*q[3, :] - q[0, :]*q[2, :],
                      q[0, :]*q[1, :] + q[2, :]*q[3, :],
                      q[0, :]**2 + q[3, :]**2 - 0.5))
    return S2


def _classS2(X, Q):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    distances, IND = nbrs.kneighbors(Q)
    NC = np.bincount(IND[:, 0].squeeze())
    return (IND, NC)


def op(q, shAngWidth, PDsizeTh, visual, thres, *fig):
    nG = int(4 * np.pi / (shAngWidth**2))
    # reference angles
    S20, it = distribute3Sphere(nG)

    S20 = S20.T
    # projection angles
    S2 = _get_S2(q)

    IND, NC = _classS2(S20.T, S2.T)

    # non-thresholded
    CG1 = [[] for i in range(S20.shape[1])]
    for i, index in enumerate(IND.ravel()):
        CG1[index].append(i)
    CG1 = np.array([np.array(a) for a in CG1], dtype=object)

    # lower-thresholded
    mid = S20.shape[1] // 2
    # halving first
    NC1 = NC[:mid]
    NC2 = NC[mid:]

    NIND = []
    if len(NC1) >= len(NC2):
        pd = 0  # PD index
        for i in NC1:  # NC1 is the occupancy of PrD=pd
            if i >= PDsizeTh:
                NIND.append(pd)
            pd += 1
    else:
        pd = mid  # PD index
        for i in NC2:  # NC2 is the occupancy of PrD=pd
            if i >= PDsizeTh:
                NIND.append(pd)
            pd += 1

    # find the "conjugate" bins
    S20_th = S20[:, NIND]

    CG = []
    for i in range(len(NIND)):
        a = (IND == NIND[i]).nonzero()[0]
        # upper-thresholded
        if len(a) > thres:
            a = a[:thres]
        CG.append(a)

    CG = np.array(CG, dtype=object)

    return (CG1, CG, nG, S2, S20_th, S20, NC, NIND)
    #CG1: list of lists, each of which is a list of image indices within one PD
    #CG: thresholded version of CG1
    #nG: approximate number of tessellated bins
    #S2: cartesian coordinates of each of particles' angular position on S2 sphere
    #S20_th: thresholded version of S20
    #S20: cartesian coordinates of each bin-center on S2 sphere
    #NC: list of occupancies of each PD
