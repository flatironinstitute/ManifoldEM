import numpy as np

from collections import namedtuple
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from ManifoldEM import slaplacianonFly
"""
%SEMBEDDING  Laplacian eigenfunction embedding using sparse arrays

"""
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''


def op(yVal, yCol, yRow, nS, options1):
    options = namedtuple('options', 'sigma alpha visual nEigs autotune')
    options.sigma = options1.sigma
    options.alpha = options1.alpha
    options.nEigs = options1.nEigs
    options.autotune = 0

    l, sigmaTune = slaplacianonFly.op(yVal, yCol, yRow, nS, options)
    try:
        vals, vecs = eigsh(l, k=options.nEigs + 1, maxiter=300)
    except ArpackNoConvergence as e:

        vals = e.eigenvalues
        vecs = e.eigenvectors
        print("eigsh not converging in 300 iterations...")
    ix = np.argsort(vals)[::-1]
    vals = np.sort(vals)[::-1]
    vecs = vecs[:, ix]

    return (vals, vecs)
