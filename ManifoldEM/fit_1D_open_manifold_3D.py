import numpy as np
# FIXME: rework this whole file

eps = 1e-4


class _Params:
    """
    A class to hold configuration parameters for an iterative fitting or optimization process.

    This class encapsulates various parameters that control the behavior of an iterative process, such as fitting a model to data or optimizing a function. It is designed to be flexible, allowing for the easy adjustment of key parameters affecting the convergence and accuracy of the process.

    Attributes
    ----------
    nDim : int
        The dimensionality of the data or parameter space. Defaults to 3.
    maxIter : int
        The maximum number of iterations to perform in the optimization or fitting process. Defaults to 100.
    delta_a_max : float
        The maximum allowed change in parameter 'a' between iterations, controlling convergence criteria. Defaults to 1.0.
    delta_b_max : float
        The maximum allowed change in parameter 'b' between iterations, controlling convergence criteria. Defaults to 1.0.
    delta_tau_max : float
        The maximum allowed change in parameter 'tau' between iterations, controlling convergence criteria. Defaults to 0.01.
    """

    # FIXME: missing doc for p/x/x_fit
    def __init__(self):
        self.a: np.ndarray = None
        self.b: np.ndarray = None
        self.nDim: int = 3
        self.maxIter: int = 100
        self.delta_a_max: float = 1.0
        self.delta_b_max: float = 1.0
        self.delta_tau_max: float = 0.01
        self.p: int = None
        self.x: np.ndarray = None
        self.x_fit = None


def _R_p(tau_p, a):
    # global p, nDim, a, b, x
    jj = np.arange(1, a.nDim + 1)
    j_pi_tau_p = tau_p * jj * np.pi
    a_cos_j_pi_tau_p = a.a * np.cos(j_pi_tau_p)
    err = a.x[a.p, :] - a.b - a_cos_j_pi_tau_p
    fval = np.sum(err**2, axis=1)

    return fval


def _solve_d_R_d_tau_p_3D(a: _Params):
    """
    function tau=solve_d_R_d_tau_p_3D()
    %
    % tau = solve_d_R_d_tau_p_3D()
    %
    % returns the value of tau for data point p that minimizes the residual
    % R to the model x_ij = a_j cos(j*pi*tau_i) + b_j for i = p.
    %
    % j goes from 1 to 3 (this is only for 3D systems).
    %
    % i goes from 1 to nS where nS is the number of data points to be fitted.
    %
    % For a fixed set of a_j and b_j, j=1:3, tau_i for i=1:nS are
    % obtained by putting dR/d(tau_i) to zero.
    %
    % For a fixed set of tau_i, i=1:nS, a_j and b_j for j=1:3 are
    % obtained by solving 3 sets of 2x2 linear equations.
    %
    % Global input:
    %   p: index of the data point of interest.
    %   a, b: dimensions 1 x 3, the current set of model coefficients.
    %   x: dimensions nS x 3, the data points to be fitted.
    % Output:
    %   tau: tau value for data point p that best fits the model.
    %
    % copyright (c) Russell Fung 2014
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Copyright (c) Columbia University Hstau Liao 2018 (python version)

    """
    d_R_d_beta_3D = np.array(
        [
            48 * a.a[2] ** 2,
            0,
            8 * a.a[1] ** 2 - 48 * a.a[2] ** 2,
            -12 * a.a[2] * (a.x[a.p, 2] - a.b[2]),
            a.a[0] ** 2
            - 4 * a.a[1] ** 2
            + 9 * a.a[2] ** 2
            - 4 * a.a[1] * (a.x[a.p, 1] - a.b[1]),
            -a.a[0] * (a.x[a.p, 0] - a.b[0]) + 3 * a.a[2] * (a.x[a.p, 2] - a.b[2]),
        ]
    ).T
    beta = np.roots(d_R_d_beta_3D)
    beta = beta[np.abs(beta.imag) < 1e-10].real
    beta = beta[np.abs(beta) <= 1]
    tau_candidate = np.vstack((np.arccos(beta.reshape(-1, 1)) / np.pi, 0, 1))
    tau = tau_candidate[np.argmin(_R_p(tau_candidate, a))]

    return tau


def _get_fit_params(psi):
    a = _Params()
    a.maxIter = 100  # maximum number of iterations, each iteration determines
    # optimum sets of {a,b} and {\tau} in turns
    a.delta_a_max = 1  # maximum percentage change in amplitudes
    a.delta_b_max = 1  # maximum percentage change in offsets
    a.delta_tau_max = 0.01  # maximum change in values of tau

    nS = psi.shape[0]
    a.x = psi[:, 0:3]
    # (1) initial guesses for {a,b} obtained by fitting data in 2D
    # (2) initial guesses for {tau} obtained by setting d(R)/d(tau) to zero

    X = psi[:, 0]
    Z = psi[:, 2]
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X5 = X3 * X2
    X6 = X3 * X3
    sumX = np.sum(X)
    sumX2 = np.sum(X2)
    sumX3 = np.sum(X3)
    sumX4 = np.sum(X4)
    sumX5 = np.sum(X5)
    sumX6 = np.sum(X6)
    sumZ = np.sum(Z)
    sumXZ = np.dot(X.T, Z)
    sumX2Z = np.dot(X2.T, Z)
    sumX3Z = np.dot(X3.T, Z)
    A = np.array(
        [
            [sumX6, sumX5, sumX4, sumX3],
            [sumX5, sumX4, sumX3, sumX2],
            [sumX4, sumX3, sumX2, sumX],
            [sumX3, sumX2, sumX, nS],
        ]
    )
    b = np.array([sumX3Z, sumX2Z, sumXZ, sumZ])

    coeff = np.linalg.lstsq(A, b, rcond=None)[0]
    D = coeff[0]
    E = coeff[1]
    F = coeff[2]
    G = coeff[3]
    disc = E * E - 3 * D * F
    if disc < 0:
        disc = 0.0
    if np.absolute(D) < 1e-8:
        D = 1e-8
    a1 = (2.0 * np.sqrt(disc)) / (3.0 * D)
    a3 = (2.0 * disc ** (3 / 2.0)) / (27.0 * D * D)
    b1 = -E / (3 * D)
    b3 = (2.0 * E * E * E) / (27.0 * D * D) - (E * F) / (3 * D) + G

    Xb = X - 2 * b1
    Y = psi[:, 1]
    XXb = X * Xb
    X2Xb2 = XXb * XXb
    sumXXb = np.sum(XXb)
    sumX2Xb2 = np.sum(X2Xb2)
    sumY = np.sum(Y)
    sumXXbY = np.dot(XXb.T, Y)
    A = np.array([[sumX2Xb2, sumXXb], [sumXXb, nS]])
    b = np.array([sumXXbY, sumY])
    coeff = np.linalg.lstsq(A, b, rcond=None)[0]

    A = coeff[0]
    C = coeff[1]
    a2 = 2.0 * A * disc / (9.0 * D * D)
    b2 = C + (A * E * E) / (9.0 * D * D) - (2.0 * A * F) / (3.0 * D)
    a.a = np.array([a1, a2, a3])
    a.b = np.array([b1, b2, b3])

    tau = np.zeros((nS, 1))

    for a.p in range(nS):
        tau[a.p] = _solve_d_R_d_tau_p_3D(a)

    return tau, a


def fit_1D_open_manifold_3D(psi):
    """
    function [a,b,tau] = fit_1D_open_manifold_3D(psi)
   
    fit_1D_open_manifold_3D
   
    fit the eigenvectors for a 1D open manifold to the model
    x_ij = a_j cos(j*pi*tau_i) + b_j.
   
    j goes from 1 to 3 (this is only for 3D systems).
   
    i goes from 1 to nS where nS is the number of data points to be fitted.
   
    For a fixed set of a_j and b_j, j=1:3, tau_i for i=1:nS are
    obtained by putting dR/d(tau_i) to zero.
   
    For a fixed set of tau_i, i=1:nS, a_j and b_j for j=1:3 are
    obtained by solving 3 sets of 2x2 linear equations.
   
    Fit parameters and initial set of {\tau} are specified in
   
      get_fit_1D_open_manifold_3D_param.m
   
    copyright (c) Russell Fung 2014
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """
    tau, a = _get_fit_params(psi)
    nS = a.x.shape[0]

    for _ in range(a.maxIter):
        # solve for a and b
        a_old = a.a
        b_old = a.b
        j_pi_tau = np.dot(tau, np.pi * np.array([[1, 2, 3]]))
        cos_j_pi_tau = np.cos(j_pi_tau)
        A11 = np.sum(cos_j_pi_tau**2, axis=0)
        A12 = np.sum(cos_j_pi_tau, axis=0)
        A21 = A12
        A22 = nS
        x_cos_j_pi_tau = a.x * cos_j_pi_tau
        b1 = np.sum(x_cos_j_pi_tau, axis=0)
        b2 = np.sum(a.x, axis=0)
        coeff = np.zeros((2, 3))
        for qq in range(3):
            A = np.array([[A11[qq], A12[qq]], [A21[qq], A22]])
            b = np.array([b1[qq], b2[qq]])
            coeff[:, qq] = np.linalg.lstsq(A, b, rcond=None)[0]

        a.a = coeff[0, :]
        a.b = coeff[1, :]

        j_pi_tau = (
            np.dot(np.linspace(0, 1, 1000).reshape(-1, 1), np.array([[1, 2, 3]]))
            * np.pi
        )
        cos_j_pi_tau = np.cos(j_pi_tau)
        tmp = a.a * cos_j_pi_tau
        a.x_fit = tmp + a.b

        # solve for tau
        tau_old = tau
        for a.p in range(nS):
            tau[a.p] = _solve_d_R_d_tau_p_3D(a)

        # calculate the changes in fitting parameters
        delta_a = np.fabs(a.a - a_old) / (np.fabs(a.a) + eps)
        delta_b = np.fabs(a.b - b_old) / (np.fabs(a.b) + eps)
        delta_tau = np.fabs(tau - tau_old)
        delta_a = np.max(delta_a) * 100
        delta_b = np.max(delta_b) * 100
        delta_tau = np.max(delta_tau)

        if (
            (delta_a < a.delta_a_max)
            and (delta_b < a.delta_b_max)
            and (delta_tau < a.delta_tau_max)
        ):
            break

    return (a.a, a.b, tau)
