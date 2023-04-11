"""Utilities that can be defined with basically one function."""
import os
import imageio
import warnings
import numpy as np

from scipy.optimize import curve_fit, OptimizeWarning

from ManifoldEM import myio, p

warnings.simplefilter(action='ignore', category=OptimizeWarning)


def clusterAvg(clust, PrD):
    dist_file = p.get_dist_file(PrD)
    data = myio.fin1(dist_file)

    imgAll = data['imgAll']
    boxSize = np.shape(imgAll)[1]

    imgAvg = np.zeros(shape=(boxSize, boxSize), dtype=float)

    for i in clust:
        imgAvg += imgAll[i]

    return imgAvg


def L2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.
    Inputs:
        A: D x M array.
        B: D x N array.
    Returns:
        E: M x N Euclidean distances between vectors in A and B.
    Author   : Roland Bunschoten
               University of Amsterdam
               Intelligent Autonomous Systems (IAS) group
               Kruislaan 403  1098 SJ Amsterdam
               tel.(+31)20-5257524
               bunschot@wins.uva.nl
    Last Rev : Wed Oct 20 08:58:08 MET DST 1999
    Tested   : PC Matlab v5.2 and Solaris Matlab v5.3
    Copyright notice: You are free to modify, extend and distribute
       this code granted that the author of the original code is
       mentioned as the original author of the code.
    Fixed by JBT (3/18/00) to work for 1-dimensional vectors
    and to warn for imaginary numbers.  Also ensures that
    output is all real, and allows the option of forcing diagonals to
    be zero.
    Basic functionality ported to Python 2.7 by JCS (9/21/2013).

    Copyright (c) Columbia University Hstau Liao 2019
    """
    eps = 1e-8

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.matmul(a.T, b)
    tmp = aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab
    tmp[tmp < eps] = 0
    return np.sqrt(tmp)


def svdRF(A):
    # Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Copyright (c) Columbia University Hstau Liao 2018 (python version)

    def tidyUp(D, EV):
        order = np.argsort(D)[::-1]
        D = np.sort(D)[::-1]
        EV = EV[:, order]
        sqrtD = np.sqrt(D)
        S = np.diag(sqrtD)
        invS = np.diag(1. / sqrtD)
        return (D, EV, S, invS)

    D1, D2 = A.shape
    if D1 > D2:
        D, V = np.linalg.eigh(np.matmul(A.T, A))
        D, V, S, invS = tidyUp(D, V)
        U = np.matmul(A, np.matmul(V, invS))
    else:
        D, U = np.linalg.eigh(np.matmul(A, A.T))
        D, U, S, invS = tidyUp(D, U)
        V = np.matmul(A.T, np.matmul(U, invS))

    return (U, S, V)


def makeMovie(IMG1, prD, psinum, fps):
    # Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Copyright (c) Columbia University Hstau Liao 2018 (python version)
    # Copyright (c) Columbia University Evan Seitz 2019 (python version)

    dim = int(np.sqrt(max(IMG1.shape)))  # window size
    nframes = IMG1.shape[1]
    images = -IMG1
    gif_path = os.path.join(p.out_dir, "topos", f"PrD_{prD + 1}", f'psi_{psinum + 1}.gif')
    frame_dt = 1.0/fps
    with imageio.get_writer(gif_path, mode='I', duration=frame_dt) as writer:
        for i in range(nframes):
            img = images[:, i].reshape(dim, dim)
            frame = np.round(255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)

            frame_path = p.out_dir + '/topos/PrD_{}/psi_{}/frame{:02d}.png'.format(prD + 1, psinum + 1, i)
            imageio.imwrite(frame_path, frame)
            writer.append_data(frame)


def fergusonE(D, logEps, a0):
    # Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Copyright (c) Columbia University Hstau Liao 2018 (python version)
    # Copyright (c) Columbia University Evan Seitz 2019 (python version)

    def fun(xx, aa0, aa1, aa2, aa3):
        return aa3 + aa2 * np.tanh(aa0 * xx + aa1)

    def find_thres(logEps, D2):
        eps = np.exp(logEps)
        d = 1. / (2. * np.max(eps)) * D2
        sg = np.sort(d)
        ss = np.sum(np.exp(-sg))
        thr = max(-np.log(0.01 * ss / len(D2)), 10)  # taking 1% of the average (10)
        return thr

    # range of values to try:
    logSumWij = np.zeros(len(logEps))
    D2 = D * D
    thr = find_thres(logEps, D2)
    for k in range(len(logEps)):
        eps = np.exp(logEps[k])
        d = 1. / (2. * eps) * D2
        d = -d[d < thr]
        Wij = np.exp(d)  # see Coifman 2008
        logSumWij[k] = np.log(np.sum(Wij))

    # curve fitting of a tanh():
    resnorm = np.inf
    cc = 0
    while (resnorm > 100):
        cc += 1
        popt, pcov = curve_fit(fun, logEps, logSumWij, p0=a0)
        resnorm = np.sum(np.sqrt(np.fabs(np.diag(pcov))))
        a0 = 1 * (np.random.rand(4, 1) - .5)

        residuals = logSumWij - fun(logEps, popt[0], popt[1], popt[2], popt[3])
        ss_res = np.sum(residuals**2)  # residual sum of squares
        ss_tot = np.sum((logSumWij - np.mean(logSumWij))**2)  # total sum of squares
        R_squared = 1 - (ss_res / ss_tot)  # R**2-value

    return (popt, logSumWij, resnorm, R_squared)


def annularMask(a: float, b: float, N: int, M: int):
    """
    returns a N x M matrix with an annular (donut) mask of inner
    radius a and outer radius b. Pixels outside the donut or inside the hole
    are filled with 0, and the rest with 1.

    The circles with radii a and b are centered on pixel (N/2,M/2).

    Programmed December 2007, modified by Peter Schwander December 2008 (Python version by Hstau Liao 2018)
    Copyright (c) Russell Fung 2007
    """
    aSq = a * a
    bSq = b * b
    mask = np.zeros((N, M))
    for xx in range(N):
        xDist = xx - N / 2 + 1
        xDistSq = xDist * xDist
        for yy in range(M):
            yDist = yy - M / 2
            yDistSq = yDist * yDist
            rSq = xDistSq + yDistSq
            mask[xx, yy] = (rSq >= aSq) & (rSq < bSq)

    return mask


def distribute3Sphere(numPts: int):
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
    K = numPts
    A3 = 4 * np.pi  # surface area of a unit 3-sphere
    delta = np.exp(np.log(A3 / K) / 2.)
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

        delta = delta * np.exp(np.log(float(id) / K) / 2.)

    results = results[0:K, :]
    return (results, it)


def get_wiener(CTF, posPath, posPsi1, ConOrder, num):
    # Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    # Copyright (c) Columbia University Hstau Liao 2018 (python version)
    dim = CTF.shape[1]
    SNR = 5
    CTF1 = CTF[posPath[posPsi1], :, :]
    wiener_dom = np.zeros((num - ConOrder, dim, dim), dtype='float64')
    for i in range(num - ConOrder):
        for ii in range(ConOrder):
            ind_CTF = ConOrder - ii + i
            wiener_dom[i, :, :] = wiener_dom[i, :, :] + CTF1[ind_CTF, :, :]**2

    wiener_dom = wiener_dom + 1. / SNR

    return (wiener_dom, CTF1)
