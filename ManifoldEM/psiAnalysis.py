"""
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Evan Seitz 2019 (python version)
"""

import multiprocessing

import numpy as np

from functools import partial
from scipy.fftpack import fft2, ifft2
from typing import List, Union

from ManifoldEM import myio, DMembeddingII
from ManifoldEM.params import p
from ManifoldEM.core import L2_distance, svdRF, get_wiener
from ManifoldEM.fit_1D_open_manifold_3D import fit_1D_open_manifold_3D
from ManifoldEM.util import NullEmitter
import tqdm


def _corr(a, b, n, m):
    A = a[:, n]
    B = b[:, m]
    A = A - np.mean(A)
    B = B - np.mean(B)
    try:
        co = np.dot(A, B) / (np.std(A) * np.std(B))
    except RuntimeError:
        raise RuntimeError("flat image")
    return co


def _diff_corr(a, b, maxval):
    return _corr(a, b, 0, 0) + _corr(a, b, maxval, maxval) - \
        (_corr(a, b, 0, maxval) + _corr(a, b, maxval, 0))


def _NLSA(NLSAPar, DD, posPath, posPsi1, imgAll, msk2, CTF, ExtPar):
    num = NLSAPar['num']
    ConOrder = NLSAPar['ConOrder']
    k = NLSAPar['k']
    tune = NLSAPar['tune']
    nS = NLSAPar['nS']
    psiTrunc = NLSAPar['psiTrunc']

    ConD = np.zeros((num - ConOrder, num - ConOrder))
    for i in range(ConOrder):
        Ind = range(i, num - ConOrder + i)
        ConD += DD[Ind][:, Ind]

    # find the manifold mapping:
    lambdaC, psiC, _, mu, _, _, _, _ = DMembeddingII.op(ConD, k, tune, 600000)

    lambdaC = lambdaC[lambdaC > 0]  ## lambdaC not used? REVIEW
    psiC1 = np.copy(psiC)
    # rearrange arrays
    if 'prD' in ExtPar:
        IMG1 = imgAll[posPath[posPsi1], :, :]
        # Wiener filtering
        wiener_dom, CTF1 = get_wiener(CTF, posPath, posPsi1, ConOrder, num)
    elif 'cuti' in ExtPar:
        IMG1 = imgAll[posPsi1, :, :]

    dim = CTF.shape[1]
    ell = psiTrunc - 1
    N = psiC.shape[0]
    psiC = np.hstack((np.ones((N, 1)), psiC[:, 0:ell]))
    mu_psi = mu.reshape((-1, 1)) * psiC
    A = np.zeros((ConOrder * dim * dim, ell + 1), dtype='float64')
    tmp = np.zeros((dim * dim, num - ConOrder), dtype='float64')

    for ii in range(ConOrder):
        for i in range(num - ConOrder):
            ind1 = 0
            ind2 = dim * dim  #max(IMG1.shape)
            ind3 = ConOrder - ii + i - 1
            img = IMG1[ind3, :, :]
            if 'prD' in ExtPar:
                img_f = fft2(img)  #.reshape(dim, dim)) T only for matlab
                CTF_i = CTF1[ind3, :, :]
                img_f_wiener = img_f * (CTF_i / wiener_dom[i, :, :])
                img = ifft2(img_f_wiener).real
                img = img * msk2  # April 2020
            tmp[ind1:ind2, i] = np.squeeze(img.T.reshape(-1, 1))

        mm = dim * dim  #max(IMG1.shape)
        ind4 = ii * mm
        ind5 = ind4 + mm
        A[ind4:ind5, :] = np.matmul(tmp, mu_psi)

    TF = np.isreal(A)
    if TF.any() != True:
        print('A is an imaginary matrix!')
        sys.exit()

    U, S, V = svdRF(A)
    VX = np.matmul(V.T, psiC.T)

    sdiag = np.diag(S)

    Npixel = dim * dim
    Topo_mean = np.zeros((Npixel, psiTrunc))
    for ii in range(psiTrunc):  # of topos considered
        #s = s + 1  needed?
        Topo = np.ones((Npixel, ConOrder)) * np.Inf

        for k in range(ConOrder):
            Topo[:, k] = U[k * Npixel:(k + 1) * Npixel, ii]
        Topo_mean[:, ii] = np.mean(Topo, axis=1)

    # unwrapping... REVIEW; allow user option to select from a list of chronos ([0,1,3]) to retain (i.e., not just i1, i2)
    i2 = 1
    i1 = 0

    ConImgT = np.zeros((max(U.shape), ell + 1), dtype='float64')
    for i in range(i1, i2 + 1):
        # %ConImgT = U(:,i) *(sdiag(i)* V(:,i)')*psiC';
        ConImgT = ConImgT + np.matmul(U[:, i].reshape(-1, 1), sdiag[i] * (V[:, i].reshape(1, -1)))

    recNum = ConOrder
    #tmp = np.zeros((Npixel,num-ConOrder),dtype='float64')
    IMGT = np.zeros((Npixel, nS - ConOrder - recNum), dtype='float64')
    for i in range(recNum):
        ind1 = i * Npixel
        ind2 = ind1 + Npixel
        tmp = np.matmul(ConImgT[ind1:ind2, :], psiC.T)
        for ii in range(num - 2 * ConOrder):
            ind3 = i + ii
            ttmp = IMGT[:, ii]
            ttmp = ttmp + tmp[:, ind3]
            IMGT[:, ii] = ttmp

    # normalize per frame so that mean=0 std=1, whole frame (this needs justif)
    for i in range(IMGT.shape[1]):
        ttmp = IMGT[:, i]
        try:
            ttmp = (ttmp - np.mean(ttmp)) / np.std(ttmp)
        except:
            print("flat image")
            exit(0)
        IMGT[:, i] = ttmp

    nSrecon = min(IMGT.shape)
    Drecon = L2_distance(IMGT, IMGT)
    k = nSrecon

    lamb, psirec, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op((Drecon**2), k, tune, 30)

    lamb = lamb[lamb > 0]
    a, b, tau = fit_1D_open_manifold_3D(psirec)

    # tau is #part (num-2ConOrder?)
    # psirec is #part x #eigs

    if NLSAPar['save'] is True:
        myio.fout1(ExtPar['filename'], psirec=psirec, tau=tau, a=a, b=b)

    return (IMGT, Topo_mean, psirec, psiC1, sdiag, VX, mu, tau)


def psi_analysis_single(input_data, con_order_range, traj_name, is_full, psi_trunc):
    dist_file = input_data[0]
    psi_file = input_data[1]  # 15-dim diffusion map coordinates
    psi2_file = input_data[2]  # output to be generated by Psi Analysis
    EL_file = input_data[3]
    psinums = input_data[4]
    senses = input_data[5]
    prD = input_data[6]
    if len(input_data) == 8:
        psi_list = input_data[7]
    else:
        psi_list = psinums
    data_IMG = myio.fin1(dist_file)
    data_psi = myio.fin1(psi_file)

    D = np.array(data_IMG['D'])  # distance matrix
    imgAll = np.array(data_IMG['imgAll'])  # every image in PD (and dimensions): e.g., shape=(numPDs,boxSize,boxSize)

    msk2 = np.array(data_IMG['msk2'])  # April 2020, vol mask to be used after ctf has been applied

    CTF = np.array(data_IMG['CTF'])
    psi = data_psi['psi']  # coordinates of all images in 15-dim space from diffusion map: e.g., shape=(numPDs,15)
    pos_path = data_psi['posPath']  # indices of every image in PD: e.g., shape=(numPDs,); [0,1,2,...(numPDs-1)]
    nS = len(pos_path)  # number of images in PD
    con_order = nS // con_order_range
    # if ConOrder is large, noise-free 2D frames expected w/ small range of conformations, \
    # while losing snapshots at edges

    dim = int(np.sqrt(imgAll.size / D.shape[0]))
    CTF = CTF.reshape(D.shape[0], dim, dim)  # needed only if read from matlab
    pos_path = np.squeeze(pos_path)
    D = D[pos_path][:, pos_path]

    extra_params = dict(outDir='', prD=prD)
    for psinum in psi_list:  # for each reaction coordinates do the following:
        if psinum == -1:
            continue
        psi_sorted_ind = np.argsort(
            psi[:, psinum])  # e.g., shape=(numPDs,): reordering image indices along each diff map coord
        pos_psi1 = psi_sorted_ind  # duplicate of above...

        DD = D[pos_psi1]
        DD = DD[:, pos_psi1]  # distance matrix with indices of images re-arranged along current diffusion map coordinate
        num = DD.shape[1]  # number of images in PD (duplicate of nS?)
        k = num - con_order

        NLSAPar = dict(num=num, ConOrder=con_order, k=k, tune=p.tune, nS=nS, save=False, psiTrunc=psi_trunc)
        IMGT, Topo_mean, psirec, psiC1, sdiag, VX, mu, tau = _NLSA(NLSAPar, DD, pos_path, pos_psi1, imgAll, msk2, CTF,
                                                                   extra_params)

        n_s_recon = min(IMGT.shape)
        numclass = min(p.nClass, n_s_recon // 2)

        tau = (tau - min(tau)) / (max(tau) - min(tau))
        tauinds = []
        i1 = 0
        i2 = IMGT.shape[0]

        IMG1 = np.zeros((i2, numclass), dtype='float64')
        for i in range(numclass):
            ind1 = float(i) / numclass
            ind2 = ind1 + 1. / numclass
            if (i == numclass - 1):
                tauind = ((tau >= ind1) & (tau <= ind2)).nonzero()[0]
            else:
                tauind = ((tau >= ind1) & (tau < ind2)).nonzero()[0]
            while (tauind.size == 0):
                sc = 1. / (numclass * 2.)
                ind1 = ind1 - sc * ind1
                ind2 = ind2 + sc * ind2
                tauind = ((tau >= ind1) & (tau < ind2)).nonzero()[0]

            IMG1[i1:i2, i] = IMGT[:, tauind[0]]
            tauinds.append(tauind[0])
        if is_full:  # second pass for EL1D
            #  adjust tau by comparing the IMG1s
            psi2_file = '{}_psi_{}'.format(psi2_file, psinum)
            data = myio.fin1(psi2_file)
            IMG1a = data['IMG1']

            dc = _diff_corr(IMG1, IMG1a, numclass - 1)
            if (senses[0] == -1 and dc > 0) or senses[0] == 1 and dc < 0:
                tau = 1 - tau

            out_file = f'{EL_file}_{traj_name}_1'
            myio.fout1(out_file, IMG1=IMG1, IMGT=IMGT, posPath=pos_path, PosPsi1=pos_psi1, psirec=psirec,
                       tau=tau, psiC1=psiC1, mu=mu, VX=VX, sdiag=sdiag, Topo_mean=Topo_mean, tauinds=tauinds)

        else:  # first pass
            out_file = f'{psi2_file}_psi_{psinum}'
            myio.fout1(out_file, IMG1=IMG1, psirec=psirec, tau=tau, psiC1=psiC1, mu=mu, VX=VX, sdiag=sdiag,
                       Topo_mean=Topo_mean, tauinds=tauinds)


def _construct_input_data(prd_list: Union[List[int], None], N):
    ll = []
    psi_nums_all = np.tile(np.array(range(p.num_psis)), (N, 1))  # numberofJobs x num_psis
    senses_all = np.tile(np.ones(p.num_psis), (N, 1))  # numberofJobs x num_psis

    valid_prds = set(range(N))
    if prd_list is not None:
        requested_prds = set(prd_list)
        invalid_prds = requested_prds.difference(valid_prds)
        if invalid_prds:
            print(f"Warning: requested invalid prds: {invalid_prds}")
        valid_prds = valid_prds.intersection(requested_prds)

    for prD in valid_prds:
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        psi2_file = p.get_psi2_file(prD)
        EL_file = p.get_EL_file(prD)
        psinums = psi_nums_all[prD, :]
        senses = senses_all[prD, :]
        psi_list = list(range(len(psinums)))  # list of incomplete psi values per PD
        ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD, psi_list])

    return ll


def op(prd_list: Union[List[int], None], *argv):
    print("Computing the NLSA snapshots...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    input_data = _construct_input_data(prd_list, p.numberofJobs)
    n_jobs = len(input_data)
    progress3 = argv[0] if use_gui_progress else NullEmitter()
    local_psi_func = partial(psi_analysis_single,
                             con_order_range=p.conOrderRange,
                             traj_name=p.trajName,
                             is_full=0,
                             psi_trunc=p.num_psiTrunc)

    if p.ncpu == 1:
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            local_psi_func(datai)
            progress3.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(local_psi_func, input_data)),
                                  total=n_jobs,
                                  disable=use_gui_progress):
                progress3.emit(int(99 * i / n_jobs))

    p.save()
    progress3.emit(100)
