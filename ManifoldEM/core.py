"""Utilities that can be defined with basically one function."""
import os
import io
import zipfile
import imageio
import warnings
import numpy as np

from scipy.ndimage import affine_transform
from scipy.optimize import curve_fit, OptimizeWarning

from ManifoldEM import myio
from ManifoldEM.params import params
from ManifoldEM.quaternion import q2Spider

warnings.simplefilter(action='ignore', category=OptimizeWarning)


def clusterAvg(clust, PrD):
    dist_file = params.get_dist_file(PrD)
    data = myio.fin1(dist_file)

    imgAll = data['imgAll']
    boxSize = np.shape(imgAll)[1]

    imgAvg = np.zeros(shape=(boxSize, boxSize), dtype=float)

    for i in clust:
        imgAvg += imgAll[i]

    return imgAvg


def L2_distance(a, b):
    """
    Computes the Euclidean distance matrix between two sets of vectors.

    This function calculates the pairwise Euclidean (L2) distances between vectors in two sets,
    represented by matrices A and B. The computation is vectorized for efficiency, making use
    of broadcasting and matrix operations to avoid explicit loops over elements.

    Parameters
    ----------
        a : ndarray
            D x M array where D is the dimensionality of each vector and M is the number
            of vectors in the first set.
        b : ndarray
            D x N array where D is the dimensionality of each vector (matching the first set)
            and N is the number of vectors in the second set.

    Returns
    -------
    ndarray
        M x N array containing the Euclidean distances between each pair of vectors
        from the first set to the second set.

    Raises
    ------
    ValueError
        If the input matrices A and B do not have the same dimensionality (i.e., the number
        of rows D does not match).

    Notes
    -----
    - The function ensures numerical stability by setting very small negative values (which can
      arise due to floating point arithmetic errors) to zero before taking the square root.
    - This implementation assumes that both input matrices are real-valued.
    - Basic functionality ported to Python 2.7 by JCS (9/21/2013).
    - Copyright (c) Columbia University Hstau Liao 2019
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
    """
    Performs a singular value decomposition-like operation on matrix A using eigenvalue decomposition.

    This function is tailored for matrices that are not necessarily square, calculating the eigendecomposition
    of A^T*A or A*A^T as appropriate based on the shape of A. It then tidies up the eigenvalues and eigenvectors
    to align with the conventional output of SVD, providing matrices U, S, and V such that A ≈ U*S*V^T.

    Parameters
    ----------
    A : ndarray
        D1 x D2 matrix for which the SVD-like decomposition is to be performed.
        `A` can be of any shape, not necessarily square.

    Returns
    -------
    tuple
        ndarray
            An orthogonal matrix containing the left singular vectors of A.
        ndarray
            A diagonal matrix with the square roots of the eigenvalues of A^T*A or A*A^T
            along the diagonal, representing the singular values of A.
        ndarray
            An orthogonal matrix containing the right singular vectors of A.

    Details
    -------
    The function internally defines a `tidyUp` helper function to sort the eigenvalues and eigenvectors,
    and to calculate the matrices S (singular values) and invS (inverse of S). The choice of decomposing
    A^T*A or A*A^T is based on the dimensions of A to ensure computational efficiency.

    Notes
    -----
    - This function does not directly use the SVD function from numpy.linalg but achieves a similar result
      through eigenvalue decomposition, which can be more efficient or numerically stable in certain contexts.
    - Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    - Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """

    def tidyUp(D, EV):
        # Sort the eigenvalues and eigenvectors in descending order
        order = np.argsort(D)[::-1]
        D = np.sort(D)[::-1]
        EV = EV[:, order]

        # Calculate the matrices for singular values and their inverses
        sqrtD = np.sqrt(D)
        S = np.diag(sqrtD)
        invS = np.diag(1. / sqrtD)

        return (D, EV, S, invS)

    D1, D2 = A.shape
    if D1 > D2:
        # For tall matrices, decompose A^T*A
        D, V = np.linalg.eigh(np.matmul(A.T, A))
        D, V, S, invS = tidyUp(D, V)
        U = np.matmul(A, np.matmul(V, invS))
    else:
        # For wide matrices, decompose A*A^T
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
    gif_path = os.path.join(params.out_dir, "topos", f"PrD_{prD + 1}", f'psi_{psinum + 1}.gif')
    zip_path = os.path.join(params.out_dir, "topos", f"PrD_{prD + 1}", f'psi_{psinum + 1}.zip')
    frame_dt = 1000.0 / fps
    with zipfile.ZipFile(zip_path, 'w') as fzip:
        with imageio.get_writer(gif_path, mode='I', duration=frame_dt) as writer:
            movie_min, movie_max = np.min(images), np.max(images)
            for i in range(nframes):
                img = images[:, i].reshape(dim, dim)
                frame = np.round(255 * (img - movie_min) / (movie_max - movie_min)).astype(np.uint8)
                frame_path = 'frame{:02d}.png'.format(i)

                b = io.BytesIO()
                imageio.imwrite(b, frame, format='png')
                b.seek(0)
                fzip.writestr(frame_path, b.read())

                writer.append_data(frame)


def fergusonE(D, logEps, a0=None):
    """
    Fits a curve using a hyperbolic tangent function to the data provided and returns the optimized parameters.

    .. math:: f(x) = d + c * \tanh(a * x + b)

    Parameters
    ----------
    D : ndarray
        An array of distances between data points.
    logEps : ndarray
        An array of logarithmic epsilon values to be used for curve fitting.
    a0 : ndarray, default=None
        Initial guess for the parameters of the hyperbolic tangent function. If None, a default value of ones(4) is used.

    Returns
    -------
    tuple
        ndarray
            Optimal values for the parameters so that the sum of the squared residuals
            of fun(xdata, *popt) - ydata is minimized.
        ndarray
            Logarithm of the sum of weighted distances for each logEps value.
        float
            The sum of the square roots of the absolute values of the diagonal of the
            covariance matrix of the parameters.
        R_squared (float): Coefficient of determination, indicating the proportion of the variance in
          the dependent variable that is predictable from the independent variable(s).

    Notes:
    - The function internally defines a `fun` function representing a hyperbolic tangent model and a
      `find_thres` function to calculate a threshold for weighting the data points.
    - The curve fitting process iterates until the residual norm (resnorm) is less than 100, adjusting
      the initial guess for the parameters (a0) in each iteration if necessary.
    - This function uses scipy's curve_fit method, which may not converge to a solution; in such cases,
      it prints the residual norm, the parameters attempted, and the error identifier (ier).
    - Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    - Copyright (c) Columbia University Hstau Liao 2018 (python version)
    - Copyright (c) Columbia University Evan Seitz 2019 (python version)
    """

    if a0 is None:
        a0 = np.ones(4)

    def fun(x, a, b, c, d):
        return d + c * np.tanh(a * x + b)

    def find_thres(logEps, D2):
        d = 0.5 * D2 / np.exp(np.max(logEps))
        ss = np.sum(np.exp(-d))
        return max(-np.log(0.01 * ss / len(D2)), 10)  # taking 1% of the average (10)

    # range of values to try:
    logSumWij = np.empty_like(logEps)
    D2 = D * D
    thr = find_thres(logEps, D2)
    for k, le in enumerate(logEps):
        d = 0.5 * D2 / np.exp(le)
        Wij = np.exp(-d[d < thr])  # see Coifman 2008
        logSumWij[k] = np.log(np.sum(Wij))

    # curve fitting of a tanh():
    resnorm = np.inf
    while (resnorm > 100):
        popt, pcov, infodict, mesg, ier = curve_fit(fun, logEps, logSumWij, p0=a0, full_output=True)
        resnorm = np.sum(np.sqrt(np.fabs(np.diag(pcov))))
        if ier < 1 or ier > 4:
            print(resnorm, popt, ier)
        a0 *= 0.5

        residuals = logSumWij - fun(logEps, *popt)
        ss_res = np.sum(residuals**2)  # residual sum of squares
        ss_tot = np.sum((logSumWij - np.mean(logSumWij))**2)  # total sum of squares
        R_squared = 1 - (ss_res / ss_tot)  # R**2-value

    return (popt, logSumWij, resnorm, R_squared)


def annular_mask(a: float, b: float, N: int, M: int):
    """
    Generates an N x M matrix representing an annular (donut-shaped) mask.

    Parameters
    ----------
    a : float
        The inner radius of the annulus.
    b : float
        The outer radius of the annulus.
    N : int
        The number of rows in the output matrix.
    M : int
        The number of columns in the output matrix.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, M) where pixels within the annular region
        are marked with 1, and all other pixels are marked with 0.

    Notes
    -----
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


def get_wiener(CTF, posPath, posPsi1, ConOrder, num):
    """
    Computes the Wiener filter domain for a set of CTF values.

    This function calculates the Wiener filter domain based on the given CTF values, taking into account
    the position path, the position in the Psi coordinate, the continuity order, and the total number of
    elements. It is particularly useful in cryo-EM image processing for deconvolving images with known
    CTF distortions under a specified signal-to-noise ratio (SNR).

    Parameters
    ----------
    CTF : ndarray
        A 3D array of CTF values with shape (N, dim, dim), where N is the number of CTF
        patterns, and 'dim' is the dimensionality of each CTF pattern.
    posPath : ndarray
        An array of indices specifying the order in which CTF patterns are considered
        in the analysis.
    posPsi1 : int
        The position index within the Psi coordinate, indicating the specific CTF pattern
        to start with in the computation.
    ConOrder : int
        The continuity order, specifying how many adjacent CTF patterns to consider for
        averaging in the Wiener domain calculation.
    num : int
        The total number of elements or CTF patterns to include in the computation from the
        arting position `posPsi1`.

    Returns
    -------
    tuple
        ndarray
            The computed Wiener filter domain, a 3D array with shape
            (num - ConOrder, dim, dim), representing the filtered domain for
            each considered CTF pattern.
        ndarray
            The subset of CTF patterns used in the Wiener domain calculation,
            with shape (num - ConOrder, dim, dim).

    Notes
    -----
    - The function assumes a constant signal-to-noise ratio (SNR) for the simplification of the Wiener
      filter calculation. The SNR is hardcoded as 5, but this value can be adjusted based on specific
      requirements or experimental data.
    - The Wiener filter domain is calculated by summing the squares of the selected CTF patterns and
      then adjusting for the SNR, following the principles of Wiener deconvolution in the frequency domain.
    - Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    - Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """
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


def euler_rot_matrix_3D_spider(Phi, Theta, Psi):
    """
    This function calculates the rotation matrix for a given set of Euler angles
    following the SPIDER convention.

    Parameters
    ----------
    Phi : float
        Euler angle phi
    Theta : float
        Euler angle theta
    Psi : float
        Euler angle psi

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.
    """
    R = np.array([
        [
            np.cos(Phi) * np.cos(Psi) * np.cos(Theta) + (-1) * np.sin(Phi) * np.sin(Psi),
            np.cos(Psi) * np.cos(Theta) * np.sin(Phi) + np.cos(Phi) * np.sin(Psi),
            (-1) * np.cos(Psi) * np.sin(Theta)
        ],
        [
            (-1) * np.cos(Psi) * np.sin(Phi) + (-1) * np.cos(Phi) * np.cos(Theta) * np.sin(Psi),
            np.cos(Phi) * np.cos(Psi) + (-1) * np.cos(Theta) * np.sin(Phi) * np.sin(Psi),
            np.sin(Psi) * np.sin(Theta)
        ],
        [
            np.cos(Phi) * np.sin(Theta),
            np.sin(Phi) * np.sin(Theta),
            np.cos(Theta)
        ]
    ])

    return R


## this usage of affine transform function with separate rotation and translation (offset)
def rotate_volume_euler(vol, sym):
    """
    Rotates a 3D volume using Euler angles.

    Parameters
    ----------
    vol : np.ndarray
        The input 3D volume to be rotated.
    sym : list or np.ndarray
        Euler angles [Phi, Theta, Psi] for the rotation.

    Returns
    -------
    np.ndarray
        The rotated 3D volume.

    Notes
    -----
    The function computes the rotation matrix from the Euler angles and applies it to the input volume.
    The 'nearest' mode is used for interpolation during the affine transformation.
    """
    dims = vol.shape
    rotmat = euler_rot_matrix_3D_spider(sym[2], sym[1], sym[0])

    # if input euler angles are not already negative, then we have to take the inverse.
    T_inv = rotmat

    c_in = 0.5 * np.array(dims)
    c_out = 0.5 * np.array(dims)
    cen_offset = c_in - np.dot(T_inv, c_out)
    rho = affine_transform(input=vol, matrix=T_inv, offset=cen_offset, output_shape=dims, mode='nearest')
    return rho


def get_euler_from_PD(PD):
    """
    Converts projection direction (PD) to Euler angles and the associated quaternion.

    Parameters
    ----------
    PD : np.ndarray
        The projection direction (unit vector array (x, y, z)).

    Returns
    -------
    np.ndarray
        Euler angles [Phi, Theta, 0.0] (Psi is degenerate, so set to zero)
    """
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0]).T
    q1 = Qr / np.sqrt(sum(Qr**2))

    phi, theta, _ = q2Spider(q1)
    return np.array([phi, theta, 0.0])


def project_mask(vol, PD):
    """
    Projects a 3D volume onto a 2D plane using a given projection direction (PD) and generates a mask.

    The function rotates the volume according to the projection direction, sums up the intensity
    along the z-axis to create a 2D projection, and then thresholds the projection to create a binary mask.

    Parameters
    ----------
    vol : np.ndarray
        The input 3D volume.
    PD : np.ndarray
        The projection direction: unit vector (x, y, z).

    Returns
    -------
    np.ndarray
        A 2D mask generated from the projected volume.

    Note:
    - This function relies on `np.swapaxes` to rearrange the axes of the volume for proper orientation.
    - `getEuler_from_PD` is an external function that converts the projection direction into Euler angles.
      This function is assumed to be defined elsewhere and is crucial for determining the rotation needed.
    - `rotateVolumeEuler` applies the rotation to the volume. This function computes the rotation matrix
      from Euler angles and uses `affine_transform` from `scipy.ndimage` for the rotation.
    - The rotation is based on the inverse transformation principle, where the volume is rotated in the opposite
      direction of the given Euler angles to simulate the projection from that direction.
    - After rotation, the volume is summed along the z-axis to create a 2D projection. The resulting projection
      is then thresholded to generate a binary mask, where pixels with values greater than 1 are set to True.
    """

    vol = np.swapaxes(vol, 0, 2)
    nPix = vol.shape[0]

    sym = get_euler_from_PD(PD)
    sym = sym * (-1.)  # for inverse transformation

    rho = rotate_volume_euler(vol, sym)

    msk = np.sum(rho, axis=2)  # axis= 2 is z slice after swapping axes(0,2)
    msk = msk.reshape(nPix, nPix).T
    msk = msk > 1
    return msk
