import copy
from functools import partial
import mrcfile
import multiprocessing
import os
import pandas
import tqdm

import numpy as np

from ManifoldEM import myio, util, quaternion, star
from ManifoldEM.util import NullEmitter
from ManifoldEM.params import params
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''

def extract_traj_data_by_prd(prds_filename, trajTaus, tauAvg, posPathAll, posPsi1All, pathw):
    prds, traj_bin_file = prds_filename
    imgss = [[] for _ in range(params.states_per_coord)]
    phis = [[] for _ in range(params.states_per_coord)]
    thetas = [[] for _ in range(params.states_per_coord)]
    psis = [[] for _ in range(params.states_per_coord)]

    for x in prds:
        fname = params.get_EL_file(x)
        IMGT = myio.fin1(fname)['IMGT']

        dist_file = params.get_dist_file(x)
        q = myio.fin1(dist_file)['q']

        posPath = posPathAll[x]
        psi1Path = posPsi1All[x]
        q = q[:, posPath[psi1Path]]
        nS = q.shape[1]

        conOrder = nS // params.con_order_range
        q = q[:, conOrder - 1:nS - conOrder]

        # scale and flip here. IMGT is now num_images x dim^2
        IMGT = (IMGT / conOrder).T

        tau = trajTaus[x]
        tauEq = util.hist_match(tau, tauAvg)

        for bin in range(params.states_per_coord - pathw + 1):
            if bin == params.states_per_coord - pathw:
                tauBin = ((tauEq >= (float(bin) / params.states_per_coord)) &
                          (tauEq <= (bin + float(pathw)) / params.states_per_coord)).nonzero()[0]
            else:
                tauBin = ((tauEq >= (float(bin) / params.states_per_coord)) &
                          (tauEq < (bin + float(pathw)) / params.states_per_coord)).nonzero()[0]

            if not len(tauBin):
                continue

            imgs = IMGT[tauBin, :].astype(np.float32)
            qs = q[:, tauBin]
            nT = len(tauBin)
            PDs = quaternion.calc_avg_pd(qs, nT)
            phi = np.empty(nT)
            theta = np.empty(nT)
            psi = np.empty(nT)

            for offset in range(nT):
                PD = PDs[:, offset]
                phi[offset], theta[offset], psi[offset] = quaternion.psi_ang(PD)
            dim = int(np.sqrt(imgs.shape[1]))
            imgs = imgs.reshape(nT, dim, dim)  # flip here
            imgs = imgs.transpose(0, 2, 1)  # flip here

            imgss[bin].append(imgs)  # append here
            phis[bin].append(phi)
            thetas[bin].append(theta)
            psis[bin].append(psi)

    myio.fout1(traj_bin_file, imgss=imgss, phis=phis, thetas=thetas, psis=psis)


def concatenate_bin(i_bin, numberOfJobs, batch_size):
    for istart in range(0, numberOfJobs, batch_size):
        numNext = min(numberOfJobs, istart + batch_size)

        traj_bin_file = "{}name{}_group_{}_{}.pkl".format(params.traj_file, params.traj_name, istart, numNext - 1)

        data = myio.fin1(traj_bin_file)
        imgss_bin_g = data['imgss']
        phis_bin_g = data['phis']
        thetas_bin_g = data['thetas']
        psis_bin_g = data['psis']

        for x in range(istart, numNext):
            y = np.mod(x, batch_size)
            if y >= len(imgss_bin_g[i_bin]):
                continue

            if istart == 0 and x == 0:
                imgs = copy.deepcopy(imgss_bin_g[i_bin][y])
                phi = copy.deepcopy(phis_bin_g[i_bin][y])
                theta = copy.deepcopy(thetas_bin_g[i_bin][y])
                psi = copy.deepcopy(psis_bin_g[i_bin][y])
            else:
                # reuse var names
                imgs = np.concatenate([imgs, imgss_bin_g[i_bin][y]])
                phi = np.concatenate([phi, phis_bin_g[i_bin][y]])
                theta = np.concatenate([theta, thetas_bin_g[i_bin][y]])
                psi = np.concatenate([psi, psis_bin_g[i_bin][y]])

    if not len(imgss_bin_g[i_bin]):
        return

    traj_file_rel = 'imgsRELION_{}_{}_of_{}.mrcs'.format(params.traj_name, i_bin + 1, params.states_per_coord)
    traj_file = os.path.join(params.bin_dir, traj_file_rel)
    ang_file = os.path.join(params.bin_dir, f'EulerAngles_{params.traj_name}_{i_bin + 1}_of_{params.states_per_coord}.star')

    if os.path.exists(traj_file):
        mrc = mrcfile.open(traj_file, mode='r+')
    else:
        mrc = mrcfile.new(traj_file)
    mrc.set_data(imgs * -1)
    mrc.close()

    df = pandas.DataFrame(data=dict(phi=phi, theta=theta, psi=psi))
    star.write_star(ang_file, traj_file_rel, df)


def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    multiprocessing.set_start_method('fork', force=True)

    numberOfJobs = len(xSelect)
    batch_size = 20
    pathw = params.width_1D
    xSelect = np.array(xSelect)
    use_gui_progress = len(argv) > 0
    progress = argv[0] if use_gui_progress else NullEmitter()

    print('Extracting and writing individual trajectory data from selected projection directions...')
    extractor = partial(extract_traj_data_by_prd,
                        trajTaus=trajTaus,
                        tauAvg=tauAvg,
                        posPathAll=posPathAll,
                        posPsi1All=posPsi1All,
                        pathw=pathw)
    input_data = []
    for istart in range(0, numberOfJobs, batch_size):
        numNext = min(numberOfJobs, istart + batch_size)
        xSel = xSelect[istart:numNext]
        traj_bin_file = f"{params.traj_file}name{params.traj_name}_group_{istart}_{numNext - 1}.pkl"
        input_data.append((xSel, traj_bin_file))

    with multiprocessing.Pool(processes=params.ncpu) as pool:
        for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(extractor, input_data)),
                              total=len(input_data),
                              disable=use_gui_progress):
            progress.emit(int(49 * i / len(input_data)))

    progress.emit(50)

    print('Concatenating prd trajectory data into bins')
    concatenator = partial(concatenate_bin, numberOfJobs=numberOfJobs, batch_size=batch_size)
    input_data = range(0, params.states_per_coord - pathw + 1)
    with multiprocessing.Pool(processes=params.ncpu) as pool:
        for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(concatenator, input_data)),
                              total=len(input_data),
                              disable=use_gui_progress):
            progress.emit(int(50 + 49 * i / len(input_data)))

    progress.emit(100)
