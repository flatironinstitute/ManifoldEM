import numpy as np

from collections import namedtuple
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from ManifoldEM.params import params
from ManifoldEM.core import fergusonE
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)

'''

def sembedding(yVal, yCol, yRow, nS, options1):
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
    options1 : namedtuple
        A namedtuple containing the options for the embedding. Expected fields
        are sigma (float), alpha (float), visual (bool), nEigs (int), and autotune (int).

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
    options = namedtuple('options', 'sigma alpha visual nEigs autotune')
    options.sigma = options1.sigma
    options.alpha = options1.alpha
    options.nEigs = options1.nEigs
    options.autotune = 0

    l, sigmaTune = slaplacian(yVal, yCol, yRow, nS, options)
    try:
        vals, vecs = eigsh(l, k=options.nEigs + 1, maxiter=300, v0=np.ones(nS), return_eigenvectors=True)
    except ArpackNoConvergence as e:
        vals = e.eigenvalues
        vecs = e.eigenvectors
        print("eigsh not converging in 300 iterations...")

    ix = np.argsort(vals)[::-1]
    vals = vals[ix]
    vecs = vecs[:, ix]

    return (vals, vecs)


def slaplacian(*arg):
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

    nAutotune : number of nearest neighbors for autotuning. Set to zero if no
    autotuning is to be performed

    sigma: width of the Gaussian kernel

    Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
    Copyright (c) Columbia University Hstau Liao 2019 (python version)
    Copyright (c) Columbia University Evan Seitz 2019 (python version)
    """
    yVal = arg[0]
    yCol = arg[1]
    yRow = arg[2]
    nS = arg[3]  #dataset size
    options = arg[4]  #options.sigma: Gaussian width

    nNZ = len(yVal)  #number of nonzero elements

    # if required, compute autotuning distances:
    if options.autotune > 0:
        print('Autotuning is not implemented in this version of slaplacian' + '\n')
    else:
        sigmaTune = options.sigma


    yVal = yVal / sigmaTune**2

    # compute the unnormalized weight matrix:
    yVal = np.exp(-yVal)  #apply exponential weights (yVal is distance**2)
    l = csc_matrix((yVal, (yRow, yCol)), shape=(nS, nS))
    d = np.array(l.sum(axis=0)).T

    if options.alpha != 1:  #apply non-isotropic normalization
        d = d**options.alpha

    yVal = yVal / (d[yRow].flatten('C') * d[yCol].flatten('C'))
    l = csc_matrix((yVal, (yRow, yCol)), shape=(nS, nS))

    # normalize by the degree matrix to form normalized graph Laplacian:
    d = np.array(l.sum(axis=0))
    d = np.sqrt(d).T

    yVal = yVal / (d[yRow].flatten('C') * d[yCol].flatten('C'))
    l = csc_matrix((yVal, (yRow, yCol)), shape=(nS, nS))
    l = np.abs(l + l.T) / 2.0  #iron out numerical wrinkles
    temp = l - l.T

    return (l, sigmaTune)


def get_yColVal(input_params):
    """
    Processes and updates arrays for values and column indices in a sparse matrix representation.

    This function takes a set of parameters related to the sparse matrix construction, including
    arrays of values and indices, and updates these arrays based on the provided batch of data.
    It is typically used in the context of constructing or updating a sparse representation of a
    graph or matrix, especially when dealing with large datasets that require batching.

    Parameters
    ----------
    params : tuple
        ndarray
            The values of the non-zero elements in the sparse matrix.
        ndarray[int]
            The original values from which yVal will be updated.
        ndarray
            The column indices of the non-zero elements in the sparse matrix.
        ndarray
            The original indices from which yCol will be updated.
        int
            The batch size, indicating the number of data points processed in this batch.
        int
            The number of nearest neighbors considered for each data point.
        int
            The initial number of nearest neighbors before filtering.
        int
            The start index of the current batch.
        int
            The end index of the current batch.
        int
            The start index for updating yVal and yCol.
        int
            The end index for updating yVal and yCol.
        int
            The current batch number.

    Returns
    -------
    tuple
        ndarray[int] : The updated column indices for the sparse matrix.
        ndarray : The updated values for the sparse matrix.

    Notes:
    - The function reshapes and filters the input arrays yVal1 and yInd1 based on the batch
      information and the number of nearest neighbors. It then updates the yVal and yCol arrays
      with the processed data for the current batch.
    - This function is part of a larger process of constructing or updating sparse matrices and
      is designed to handle data in batches for efficiency and scalability.
    """

    yVal = input_params[0]
    yVal1 = input_params[1]
    yCol = input_params[2]
    yInd1 = input_params[3]
    nB = input_params[4]
    nN = input_params[5]
    nNIn = input_params[6]
    jStart = input_params[7]
    jEnd = input_params[8]
    indStart = input_params[9]
    indEnd = input_params[10]
    iBatch = input_params[11]

    DataBatch = yVal1
    DataBatch = DataBatch.reshape(nB, nNIn).T
    DataBatch = DataBatch[:nN, :]
    DataBatch[0, :] = 0
    yVal[indStart:indEnd] = DataBatch.reshape(nN * nB, 1)
    DataBatch = yInd1
    DataBatch = DataBatch.reshape(nB, nNIn).T
    DataBatch = DataBatch[:nN, :]
    yCol[indStart:indEnd] = DataBatch.reshape(nN * nB, 1).astype(float)

    return (yCol, yVal)


def initialize(nS, nN, D):
    """
    Initializes the arrays of indices and values for constructing a sparse matrix
    representation of distances between data points.

    This function processes a distance matrix to identify the nearest neighbors for each
    data point. It then creates arrays that store the indices of these neighbors and the
    corresponding distance values. The first distance value for each data point is set to
    zero to indicate self-distance, ensuring the diagonal of the distance matrix is zero.

    Parameters
    ----------
    nS : int
        The number of samples or data points.
    nN : int
        The number of nearest neighbors to consider for each data point.
    D : ndarray
        A square distance matrix of shape (nS, nS) where D[i, j] represents
        the distance between the i-th and j-th data points.

    Returns
    tuple
        ndarray
            A flattened array of indices of the nearest neighbors for each data point.
        yVal1 : ndarray
            A flattened array of the corresponding distance values to the nearest neighbors.

    Notes:
    - The function modifies the input distance matrix D by setting the diagonal elements to
      negative infinity to ensure that each data point's self-distance does not affect the
      nearest neighbors' calculation.
    - After identifying the nearest neighbors and their distances, the function resets the
      first distance value for each data point to zero, effectively ignoring self-distance
      in the sparse matrix representation.
    """
    yInd1 = np.zeros((nN, nS), dtype='int32')
    yVal1 = np.zeros((nN, nS), dtype='float64')

    for iS in range(nS):
        D[iS, iS] = -np.Inf  # force this distance to be the minimal value
        B = np.sort(D[:, iS])
        IX = np.argsort(D[:, iS])
        yInd1[:, iS] = IX[:nN]
        yVal1[:, iS] = B[:nN]
        yVal1[0, iS] = 0  # set this distance back to zero

    yInd1 = yInd1.flatten('F')
    yVal1 = yVal1.flatten('F')
    return (yInd1, yVal1)


def construct_matrix0(Row, Col, Val, nS):
    """
    Constructs a symmetric matrix from given row indices, column indices, and values,
    specifically designed for handling squared distances.

    This function first creates a sparse matrix from the given row indices, column indices,
    and values. It then converts this sparse matrix to a dense array and performs operations
    to ensure that the resulting matrix is symmetric and represents squared distances
    correctly.

    Parameters
    ----------
    Row : ndarray
        An array of row indices for the non-zero elements in the matrix.
    Col : ndarray
        An array of column indices for the non-zero elements in the matrix.
    Val : ndarray
        An array of values corresponding to the non-zero elements in the matrix.
    nS : int
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

    y = csr_matrix((Val, (Row, Col)), shape=(nS, nS)).toarray()
    y2 = y * y.T  #y2 contains the squares of the distances
    y = y**2
    y = y + y.T - y2
    return y


def construct_matrix1(Row, Col, Val, nS):
    """
    Constructs a symmetric matrix from given row indices, column indices, and values.
    This version simplifies the process by directly adjusting for symmetry without explicitly
    squaring the matrix values.

    Parameters
    ----------
    Row : ndarray
        An array of row indices for the non-zero elements in the matrix.
    Col : ndarray
        An array of column indices for the non-zero elements in the matrix.
    Val : ndarray
        An array of values corresponding to the non-zero elements in the matrix.
    nS : int
        The size of the square matrix to be constructed, indicating both the number
        of rows and columns.

    Returns
    -------
    ndarray
        A symmetric matrix constructed from the input parameters. The symmetry is
        achieved by adding the matrix to its transpose and then subtracting the element-wise
        product of the matrix and its transpose.

    Notes:
    - The function first constructs a sparse CSR (Compressed Sparse Row) matrix from the input
      indices and values. It then converts this sparse matrix to a dense array.
    - It ensures the symmetry of the resulting matrix by adding it to its transpose and
      subtracting the element-wise product of the matrix and its transpose, which contains
      the squares of the original distances. This operation corrects for any asymmetries
      that might arise from the input data or the sparse matrix construction process.
    """
    y = csr_matrix((Val, (Row, Col)), shape=(nS, nS)).toarray()
    y2 = y * y.T  #y2 contains the squares of the distances
    y = y + y.T - y2
    return y


def op(D, k, tune, prefsigma):  #*arg
    nS = D.shape[0]
    nN = k  #total number of entries is nS*nN
    yInd1, yVal1 = initialize(nS, nN, D)

    # diffraction patterns:
    nB = nS  #batch size (number of diff. patterns per batch)
    nNIn = k  #number of input nearest neighbors
    nN = k  #number of output nearest neighbors
    iBatch = 1  #REVIEW THIS SECTION, absurd to keep this here... nB=nS then nBatch = nS/nB ...???

    yVal = np.zeros((nS * nN, 1))
    yCol = np.zeros((nS * nN, 1))

    nBatch = int(nS / nB)
    for iBatch in range(nBatch):
        # linear indices in the non-symmetric distance matrix (indStart, indEnd)
        indStart = iBatch * nB * nN
        indEnd = (iBatch + 1) * nB * nN
        # diffraction pattern indices (jStart, jEnd):
        jStart = iBatch * nB
        jEnd = (iBatch + 1) * nB
        myparams = (yVal, yVal1, yCol, yInd1, nB, nN, nNIn, jStart, jEnd, indStart, indEnd, iBatch)
        yCol, yVal = get_yColVal(myparams)

    # symmetrizing the distance matrix:
    yRow = np.ones((nN, 1)) * range(nS)
    yRow = yRow.reshape(nS * nN, 1)
    ifZero = yVal < 1e-6
    yRowNZ = yRow[~ifZero]
    yColNZ = yCol[~ifZero]
    yValNZ = np.sqrt(yVal[~ifZero])
    nNZ = len(yRowNZ)  #number of nonzero elements in the non-sym matrix
    yRow = yRow[ifZero]
    yCol = yCol[ifZero]
    nZ = len(yRow)  #number of zero elements in the non-sym matrix

    y = construct_matrix0(yRowNZ, yColNZ, yValNZ, nS)
    yRowNZ = y.nonzero()[0]
    yColNZ = y.nonzero()[1]
    yValNZ = y[y.nonzero()]
    nNZ = len(y.nonzero()[0])  #number of nonzero elements in the sym matrix

    y = construct_matrix1(yRow, yCol, np.ones((nZ, 1)).flatten(), nS)
    y = csr_matrix((np.ones((nZ, 1)).flatten(), (yRow, yCol)), shape=(nS, nS)).toarray()
    yRow = y.nonzero()[0]
    yCol = y.nonzero()[1]
    yVal = y[y.nonzero()]

    yVal[:] = 0
    nZ = len(y.nonzero()[0])  #number of zero elements in the sym matrix
    yRow = np.hstack((yRow, yRowNZ)).astype(int)
    yCol = np.hstack((yCol, yColNZ)).astype(int)
    yVal = np.hstack((yVal, yValNZ))

    count = 0
    resnorm = np.inf

    logEps = np.arange(-150, 150.2, 0.2)
    popt, logSumWij, resnorm, R_squared = fergusonE(np.sqrt(yVal), logEps)
    nS = D.shape[0]
    nEigs = min(params.num_eigs, nS - 3)  #number of eigenfunctions to compute
    nA = 0  #autotuning parameter
    nN = k  #number of nearest neighbors
    nNA = 0  #number of nearest neighbors used for autotuning
    if count < 20:
        alpha = 1  #kernel normalization
        #alpha = 1.0: Laplace-Beltrami operator
        #alpha = 0.5: Fokker-Planck diffusion
        #alpha = 0.0: graph Laplacian normalization
        sigma = tune * np.sqrt(2 * np.exp(-popt[1] / popt[0]))  #Gaussian Kernel width (=1 for autotuning)
        #sigma = np.sqrt(2 * np.exp(-popt[1] / popt[0])) #as in Fergusson paper
    else:
        print('using prefsigma...')  #does this ever happen or can we delete? REVIEW
        sigma = prefsigma
        alpha = 1

    visual = 1
    options = namedtuple('Options', 'sigma alpha visual nEigs')
    options.sigma = sigma
    options.alpha = alpha
    options.visual = visual
    options.nEigs = nEigs

    lamb, v = sembedding(yVal, yCol, yRow, nS, options)

    #psi = v[:, 1 : nEigs+1]/np.tile(v[:, 0 ].reshape((-1,1)), (1, nEigs))
    true_shape = v.shape[1] - 1
    psi = np.zeros((v.shape[0], nEigs))
    psi[:, :true_shape] = v[:, 1:] / np.tile(v[:, 0].reshape((-1, 1)), (1, true_shape))  # could be fewer than nEigs

    ##################################
    # the Riemannian measure. Nov 2012
    mu = v[:, 0]
    mu = mu * mu  #note: sum(mu)=1
    ##################################

    return (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared)
