"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
"""

import numpy as np
import matplotlib.pyplot as plt

from ManifoldEM import myio
from ManifoldEM.DMembeddingII import diffusion_map_embedding


def get_psiPath(psi, rad, plotNum):
    """
    Identifies the indices of points in the embedded space within a specified radius from the origin.

    Parameters
    ----------
    psi : ndarray
        The matrix of embedded coordinates, where each column represents a dimension.
    rad : float
        The radius within which to consider points as being part of the path.
    plotNum : int
        The starting dimension in the embedded space for calculating distances.

    Returns
    -------
    ndarray
        An array of indices of points within the specified radius from the origin in the
        embedded space defined by the dimensions starting at plotNum.

    Notes:
    - This function calculates the Euclidean distance from the origin in a 3-dimensional space
      defined by the dimensions plotNum, plotNum+1, and plotNum+2 of the embedded coordinates.
    - Points with a distance less than 'rad' from the origin are considered part of the path.
    """
    psinum1 = plotNum
    psinum2 = plotNum + 1
    psinum3 = plotNum + 2

    psiDist = np.sqrt(
        psi[:, psinum1] ** 2 + psi[:, psinum2] ** 2 + psi[:, psinum3] ** 2
    )

    posPathInt = (psiDist < rad).nonzero()[0]

    return posPathInt


def show_plot(lamb, psi, string):
    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.bar(range(len(lamb) - 1), lamb[1:], color="green")
    ax1.scatter(psi[:, 0], psi[:, 1], marker="+", s=30, alpha=0.6)
    ax1.scatter(psi[:, 2], psi[:, 3], marker="+", s=30, alpha=0.6)
    plt.title(string, fontsize=20)
    plt.show()


def op(input_data, posPath, tune, rad, visual, doSave):
    """
    Performs manifold learning and trimming based on spectral embedding, iteratively refining
    the embedding by trimming points outside a specified radius.

    Parameters
    ----------
    input_data : tuple
        str : Path to the file containing the distance matrix.
        str : Path to the file where the psi (embedding coordinates) will be saved.
        str : Path to the file where the eigenvalues will be saved.
        int : An identifier for the current projection direction.
    posPath : Union[int, np.ndarray[int]]
        Initial positions/path of points to consider. If 0, all points are considered.
    tune : float
        A tuning parameter for adjusting the Gaussian kernel width in the embedding process.
    rad : float
        The radius within which points are considered part of the manifold.
    visual : bool
        A flag indicating whether to visualize the embedding and trimming process.
    doSave : dict
        A dictionary indicating whether to save the results ('Is' key) and the output file path.

    Notes
    -----
    - The function starts by loading the distance matrix and optionally filters it based on `posPath`.
    - It then performs spectral embedding using `DMembeddingIIop` and trims points outside the specified `rad`.
    - This process is repeated until the set of points within the radius no longer changes significantly.
    - Visualization of the embedding before and after trimming is optional and controlled by the `visual` flag.
    - Results, including the final set of points, embedding coordinates, and eigenvalues, can be saved to files.
    """
    dist_file = input_data[0]
    psi_file = input_data[1]
    eig_file = input_data[2]
    prD = input_data[3]
    data = myio.fin1(dist_file)
    D = data["D"]
    ind = data["ind"]
    nS = D.shape[1]
    if posPath == 0:
        posPath = np.arange(nS)
    D = D[posPath][:, posPath]
    nS = D.shape[1]
    k = nS

    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = diffusion_map_embedding(
        D, k, tune
    )
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
        lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = diffusion_map_embedding(
            D1, k, tune
        )
        lamb = lamb[lamb > 0]
        posPathInt = get_psiPath(psi, rad, 0)
        posPath1 = posPath1[posPathInt]

        if visual:
            show_plot(lamb, psi, "in loop")
    if visual:
        show_plot(lamb, psi, "out loop")

    posPath = posPath[posPath1]

    if doSave["Is"]:
        myio.fout1(
            psi_file,
            lamb=lamb,
            psi=psi,
            sigma=sigma,
            mu=mu,
            posPath=posPath,
            ind=ind,
            logEps=logEps,
            logSumWij=logSumWij,
            popt=popt,
            R_squared=R_squared,
        )

    with open(eig_file, "w") as file:
        for i in range(1, len(lamb)):
            file.write("%d\t%.5f\n" % (i, lamb[i]))
