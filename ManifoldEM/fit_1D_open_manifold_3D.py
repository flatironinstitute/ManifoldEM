import numpy as np

from ManifoldEM import get_fit_1D_open_manifold_3D_param, solve_d_R_d_tau_p_3D, a
'''
function [a,b,tau] = fit_1D_open_manifold_3D(psi)
% 
% fit_1D_open_manifold_3D
% 
% fit the eigenvectors for a 1D open manifold to the model
% x_ij = a_j cos(j*pi*tau_i) + b_j.
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
% Fit parameters and initial set of {\tau} are specified in
% 
%   get_fit_1D_open_manifold_3D_param.m
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    

  global p nDim a b x x_fit

'''

eps = 1e-4


def op(psi):
    #global p, nDim, a, b, x, x_fit
    a.init()
    a.nDim = 3

    tau = get_fit_1D_open_manifold_3D_param.op(psi)

    aux = np.zeros((tau.shape[0], 5))  #added
    nS = a.x.shape[0]

    for iter in range(1, a.maxIter + 1):
        '''
        #%%%%%%%%%%%%%%%%%%%%%
        #% solve for a and b %
        #%%%%%%%%%%%%%%%%%%%%%
        '''
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
            coeff[:, qq] = np.linalg.lstsq(A, b)[0]

        a.a = coeff[0, :]
        a.b = coeff[1, :]
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%
        #% plot the fitted curve %
        %%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        j_pi_tau = np.dot(np.linspace(0, 1, 1000).reshape(-1, 1), np.array([[1, 2, 3]])) * np.pi
        cos_j_pi_tau = np.cos(j_pi_tau)
        tmp = a.a * cos_j_pi_tau
        a.x_fit = tmp + a.b
        #%plot_fitted_curve(iter)
        '''
        %%%%%%%%%%%%%%%%%
        #% solve for tau %
        %%%%%%%%%%%%%%%%%
        '''
        tau_old = tau
        for a.p in range(nS):
            tau[a.p], beta = solve_d_R_d_tau_p_3D.op()  #added
            for kk in range(beta.shape[0]):
                aux[a.p, kk] = beta[kk]
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% calculate the changes in fitting parameters %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        delta_a = np.fabs(a.a - a_old) / (np.fabs(a.a) + eps)
        delta_b = np.fabs(a.b - b_old) / (np.fabs(a.b) + eps)
        delta_tau = np.fabs(tau - tau_old)
        delta_a = max(delta_a) * 100
        delta_b = max(delta_b) * 100
        delta_tau = max(delta_tau)

        if (delta_a < a.delta_a_max) and (delta_b < a.delta_b_max) and (delta_tau < a.delta_tau_max):
            break

    return (a.a, a.b, tau)
