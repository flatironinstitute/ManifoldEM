"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data

from ManifoldEM import DMembeddingII, myio
from ManifoldEM.data_store import data_store


def get_psiPath(psi, rad, plotNum):
    psinum1 = plotNum
    psinum2 = plotNum + 1
    psinum3 = plotNum + 2

    psiDist = np.sqrt(psi[:, psinum1]**2 + psi[:, psinum2]**2 + psi[:, psinum3]**2)

    posPathInt = (psiDist < rad).nonzero()[0]

    return posPathInt


def show_plot(lamb, psi, string):
    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.bar(range(len(lamb) - 1), lamb[1:], color="green")
    ax1.scatter(psi[:, 0], psi[:, 1], marker='+', s=30, alpha=0.6)
    ax1.scatter(psi[:, 2], psi[:, 3], marker='+', s=30, alpha=0.6)
    plt.title(string, fontsize=20)
    plt.show()


def op(input_data, posPath, tune, rad):
    eig_file = input_data[0]
    prd = input_data[1]

    distance_store = data_store.get_distances()
    D = distance_store.distance_matrix(prd)
    ind = distance_store.indices(prd)

    nS = D.shape[1]
    if posPath == 0:
        posPath = np.arange(nS)
    D = D[posPath][:, posPath]
    nS = D.shape[1]
    k = nS

    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op(D, k, tune, 60000)
    # lamb = lamb[lamb>0]; not used
    # psi is nS x num_psis
    # sigma is a scalar
    # mu is 1 x nS
    posPath1 = get_psiPath(psi, rad, 0)
    cc = 0
    while len(posPath1) < nS:
        cc += 1
        nS = len(posPath1)
        D1 = D[posPath1][:, posPath1]
        k = D1.shape[0]
        lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op(
            D1, k, tune, 600000)  #changed from 60000 to match Matlab
        lamb = lamb[lamb > 0]
        posPathInt = get_psiPath(psi, rad, 0)
        posPath1 = posPath1[posPathInt]

    posPath = posPath[posPath1]

    with open(eig_file, "w") as file:
        for i in range(1, len(lamb)):
            file.write("%d\t%.5f\n" % (i, lamb[i]))

    return (prd,
            dict(lamb=lamb,
                 psi=psi,
                 sigma=sigma,
                 mu=mu,
                 posPath=posPath,
                 ind=ind,
                 logEps=logEps,
                 logSumWij=logSumWij,
                 popt=popt,
                 R_squared=R_squared))
