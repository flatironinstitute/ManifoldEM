from nptyping import NDArray, Shape, Float, Int
from typing import Any
import numpy as np

from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from ManifoldEM.params import params
from ManifoldEM.core import fergusonE

"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)

"""


@dataclass
class EmbeddingOptions:
    sigma: float
    alpha: float
    n_eigs: int


def mat_from_indices(row_indices, col_indices, vals, n_rows, sparse=True):
    if sparse:
        return csr_matrix((vals, (row_indices, col_indices)), shape=(n_rows, n_rows))

    y = np.zeros((n_rows, n_rows))
    y[row_indices, col_indices] = vals
    return y


def sembedding(yVal, yCol, yRow, nS, options: EmbeddingOptions):
    """
    Laplacian eigenfunction embedding using sparse arrays.

    This function computes the eigenvalues and eigenvectors of the Laplacian matrix of a graph,
    which represents the dataset. The Laplacian matrix is constructed based on the input sparse
    matrix components (values, column indices, row pointers) and options specifying the embedding
    parameters.

    Parameters
    ----------
    yVal : ndarray
        The values of the non-zero elements in the sparse matrix representation.
    yCol : ndarray[int]
        The column indices of the non-zero elements in the sparse matrix.
    yRow : ndarray[int]
        The row indices of the non-zero elements in the sparse matrix.
    nS : int
        The number of samples or nodes in the graph.
    options : EmbeddingOptions
        A namedtuple containing the options for the embedding. Expected fields
        are sigma (float), alpha (float), and n_eigs (int).

    Returns
    -------
    tuple
        ndarray
            The computed eigenvalues of the Laplacian matrix.
        ndarray
            The computed eigenvectors of the Laplacian matrix, corresponding to the eigenvalues.

    Notes
    -----
    - The function attempts to compute the specified number of eigenvalues and eigenvectors using
      the ARPACK solver via scipy.sparse.linalg.eigsh. If the solver does not converge within the
      specified maximum number of iterations, it catches the ArpackNoConvergence exception and
      returns the eigenvalues and eigenvectors that were computed up to that point.
    - Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    - Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """
    laplacian_mat = slaplacian(yVal, yCol, yRow, nS, options)
    try:
        vals, vecs = eigsh(
            laplacian_mat,
            k=options.n_eigs + 1,
            maxiter=300,
            v0=np.ones(nS),
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as e:
        vals = e.eigenvalues
        vecs = e.eigenvectors
        print("eigsh not converging in 300 iterations...")

    ix = np.argsort(vals)[::-1]
    vals = vals[ix]
    vecs = vecs[:, ix]

    return (vals, vecs)


def slaplacian(yVal, yCol, yRow, nS, options: EmbeddingOptions):
    """
    Given a set of nS data points, and the dinstances to nN nearest neighbors
    for each data point, slaplacian computes a sparse, nY by nY symmetric
    graph Laplacian matrix l.

    The input data are supplied in the column vectors yVal and yInd of length
    nY * nN such that

    yVal( ( i - 1 ) * nN + ( 1 : nN ) ) contains the distances to the
    nN nearest neighbors of data point i sorted in ascending order, and

    yInd( ( i - 1 ) * nN + ( 1 : nN ) ) contains the indices of the nearest
    neighbors.

    yVal and yInd can be computed by calling nndist

    slaplacian admits a number of options passed as name-value pairs

    alpha : normalization, according to Coifman & Lafon

    sigma: width of the Gaussian kernel

    Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    Copyright (c) Columbia University Hstau Liao 2019 (python version)
    Copyright (c) Columbia University Evan Seitz 2019 (python version)
    """
    # compute the unnormalized weight matrix:
    laplacian_mat = mat_from_indices(
        yRow, yCol, np.exp(-yVal / options.sigma**2), nS, sparse=True
    )

    # apply non-isotropic normalization
    d = np.array(laplacian_mat.sum(axis=1)) ** options.alpha
    laplacian_mat[yRow, yCol] /= (d[yRow] * d[yCol]).flatten()
    # import ipdb
    # ipdb.set_trace()

    # normalize by the degree matrix to form normalized graph Laplacian:
    d = np.array(np.sqrt(laplacian_mat.sum(axis=1)))
    laplacian_mat[yRow, yCol] /= (d[yRow] * d[yCol]).flatten()

    # symmetrize the matrix and iron out numerical wrinkles
    laplacian_mat = np.abs(laplacian_mat + laplacian_mat.T) / 2.0

    return laplacian_mat


def sort_mat_nneighbs(D: NDArray[Shape["Any,Any"], Float], n_neighbs: int):
    """
    Initializes the arrays of indices and values for constructing a sparse matrix
    representation of distances between data points.

    This function processes a distance matrix to identify the nearest neighbors for each
    data point. It then creates arrays that store the indices of these neighbors and the
    corresponding distance values. The first distance value for each data point is set to
    zero to indicate self-distance, ensuring the diagonal of the distance matrix is zero.

    Parameters
    ----------
    D : ndarray
        A square distance matrix of shape (nS, nS) where D[i, j] represents
        the distance between the i-th and j-th data points.
    n_neighbs : int
        The number of nearest neighbors to consider for each data point.

    Returns
    tuple
        ndarray
            A flattened array of indices of the nearest neighbors for each data point.
        ndarray
            A flattened array of the corresponding distance values to the nearest neighbors.

    Notes:
    - The function modifies the input distance matrix D by setting the diagonal elements to
      negative infinity to ensure that each data point's self-distance does not affect the
      nearest neighbors' calculation.
    - After identifying the nearest neighbors and their distances, the function resets the
      first distance value for each data point to zero, effectively ignoring self-distance
      in the sparse matrix representation.
    """
    n_rows = D.shape[0]
    indices = np.zeros((n_neighbs, n_rows), dtype=np.int32)
    vals = np.zeros((n_neighbs, n_rows), dtype=np.float64)

    for iS in range(n_rows):
        D[iS, iS] = -np.Inf  # force this distance to be the minimal value
        IX = np.argsort(D[:, iS])
        indices[:, iS] = IX[:n_neighbs]
        vals[:, iS] = D[IX[:n_neighbs], iS]
        vals[0, iS] = 0  # set this distance back to zero

    return (indices.flatten(), vals.flatten())


def square_symmetrize_matrix(
    row_indices: NDArray[Shape["Any"], Int],
    col_indices: NDArray[Shape["Any"], Int],
    vals: NDArray[Shape["Any"], Float],
    n_rows: int,
) -> NDArray[Shape["Any,Any"], Float]:
    """
    Constructs a symmetric matrix from given row indices, column indices, and values,
    specifically designed for handling squared distances.

    This function first creates a sparse matrix from the given row indices, column indices,
    and values. It then converts this sparse matrix to a dense array and performs operations
    to ensure that the resulting matrix is symmetric and represents squared distances
    correctly.

    Parameters
    ----------
    row_indices : ndarray
        An array of row indices for the non-zero elements in the matrix.
    col_indices : ndarray
        An array of column indices for the non-zero elements in the matrix.
    vals : ndarray
        An array of values corresponding to the non-zero elements in the matrix.
    n_rows : int
        The size of the square matrix to be constructed, indicating both the number
        of rows and columns.

    Returns
    -------
    ndarray
        A symmetric matrix of shape [nS, nS] constructed from the input parameters, with adjustments
        to ensure correct representation of squared distances.

    Notes:
    - The function first constructs a sparse CSR (Compressed Sparse Row) matrix from the input
      indices and values. It then converts this sparse matrix to a dense array.
    - It computes the square of the dense array and its transpose to handle squared distances.
    - The final matrix is adjusted by adding the original matrix and its transpose, then
      subtracting a matrix that contains the squares of the distances, to ensure symmetry
      and correct distance representation.
    """
    y = mat_from_indices(row_indices, col_indices, vals, n_rows, False)
    y2 = y * y.T
    y = y**2
    return y + y.T - y2


def diffusion_map_embedding(D: NDArray[Shape["Any,Any"], Float], k: int, tune: float, alpha: float = 1.0):
    # alpha = 1.0: Laplace-Beltrami operator
    # alpha = 0.5: Fokker-Planck diffusion
    # alpha = 0.0: graph Laplacian normalization
    n_rows = D.shape[0]
    rows = np.tile(np.arange(n_rows), k)
    cols, vals = sort_mat_nneighbs(D, k)

    # symmetrizing the distance matrix
    val_is_zero = vals < 1e-6
    rows_nonzero, cols_nonzero = rows[~val_is_zero], cols[~val_is_zero]
    vals_nonzero = np.sqrt(vals[~val_is_zero])
    rows_zero, cols_zero = rows[val_is_zero], cols[val_is_zero]

    y = square_symmetrize_matrix(rows_nonzero, cols_nonzero, vals_nonzero, n_rows)
    rows_nonzero, cols_nonzero = y.nonzero()
    vals_nonzero = y[rows_nonzero, cols_nonzero]

    rows = np.hstack((rows_zero, rows_nonzero))
    cols = np.hstack((cols_zero, cols_nonzero))
    vals = np.hstack((np.zeros(len(rows_zero)), vals_nonzero))
    log_eps = np.linspace(-150.0, 150, 5 * 150 * 2 + 1)
    popt, log_sum_Wij, _, R_squared = fergusonE(np.sqrt(vals), log_eps)
    n_eigs = min(params.num_eigs, n_rows - 3)  # number of eigenfunctions to compute

    # Gaussian Kernel width
    sigma = tune * np.sqrt(2 * np.exp(-popt[1] / popt[0]))

    options = EmbeddingOptions(sigma=sigma, alpha=alpha, n_eigs=n_eigs)
    eig_vals, eig_vecs = sembedding(vals, cols, rows, n_rows, options)

    eig_count = eig_vecs.shape[1] - 1
    psi = np.zeros((eig_vecs.shape[0], n_eigs))
    # note that eig_count could be fewer than n_eigs
    psi[:, :eig_count] = eig_vecs[:, 1:] / eig_vecs[:, 0].reshape(-1, 1)

    # the Riemannian measure. Nov 2012
    mu = eig_vecs[:, 0]
    mu = mu * mu  # note: sum(mu)=1

    return (eig_vals, psi, sigma, mu, log_eps, log_sum_Wij, popt, R_squared)
