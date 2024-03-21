import copy
import mrcfile
import os
import pandas

import numpy as np

from ManifoldEM import myio, util, quaternion, star
from ManifoldEM.params import params
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''


def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    # S.M. June 2020
    pathw = params.width_1D

    # TODO: Have to find a way to control this from the GUI and also write extra steps to provide resume capability
    get_traj_bins = 1  # if 1, then the trajectory data is extracted from selected PDs,
    # if 0 then we skip this and read from previously saved files
    numberOfWorkers = 20  # determines how many PDs will be processed and saved together in a single pickle file
    # higer values mean overall fewer files written, but will also need more memory to write and read later.
    numberOfJobs = len(xSelect)

    # S.M. June 2020
    xSelect = np.array(xSelect)
    if get_traj_bins:
        print('Extracting and writing individual trajectory data from selected projection directions ...')

        for num in range(0, numberOfJobs, numberOfWorkers):
            imgss = [[] for i in range(params.states_per_coord)]
            phis = [[] for i in range(params.states_per_coord)]
            thetas = [[] for i in range(params.states_per_coord)]
            psis = [[] for i in range(params.states_per_coord)]

            numNext = min(numberOfJobs, num + numberOfWorkers)

            xSel = xSelect[num:numNext]
            for x in xSel:
                fname = f'{params.get_EL_file(x)}_{params.traj_name}_1'
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

            traj_bin_file = "{}name{}_group_{}_{}".format(params.traj_file, params.traj_name, num, numNext - 1)
            myio.fout1(traj_bin_file, imgss=imgss, phis=phis, thetas=thetas, psis=psis)

            print('Done saving group.')
    else:
        print('Reading previously generated trajectory data from saved projection directions...')

    # S.M. June 2020
    # loop through the nClass again and convert each list in the list to array
    for bin in range(0, params.states_per_coord - pathw + 1):
        print('\nConcatenated bin:', bin)

        for num in range(0, numberOfJobs, numberOfWorkers):
            numNext = min(numberOfJobs, num + numberOfWorkers)

            traj_bin_file = "{}name{}_group_{}_{}".format(params.traj_file, params.traj_name, num, numNext - 1)

            data = myio.fin1(traj_bin_file)
            imgss_bin_g = data['imgss']
            phis_bin_g = data['phis']
            thetas_bin_g = data['thetas']
            psis_bin_g = data['psis']

            for x in range(num, numNext):
                y = np.mod(x, numberOfWorkers)
                if y >= len(imgss_bin_g[bin]):
                    continue

                if num == 0 and x == 0:
                    imgs = copy.deepcopy(imgss_bin_g[bin][y])
                    phi = copy.deepcopy(phis_bin_g[bin][y])
                    theta = copy.deepcopy(thetas_bin_g[bin][y])
                    psi = copy.deepcopy(psis_bin_g[bin][y])
                else:
                    # reuse var names
                    imgs = np.concatenate([imgs, imgss_bin_g[bin][y]])
                    phi = np.concatenate([phi, phis_bin_g[bin][y]])
                    theta = np.concatenate([theta, thetas_bin_g[bin][y]])
                    psi = np.concatenate([psi, psis_bin_g[bin][y]])

        if not len(imgss_bin_g[bin]):
            print('Bad bin:', bin)
            continue

        print('Concatenated imgs, shape', np.shape(imgs))

        traj_file_rel = 'imgsRELION_{}_{}_of_{}.mrcs'.format(params.traj_name, bin + 1, params.states_per_coord)
        traj_file = os.path.join(params.bin_dir, traj_file_rel)
        ang_file = os.path.join(params.bin_dir, f'EulerAngles_{params.traj_name}_{bin + 1}_of_{params.states_per_coord}.star')

        if os.path.exists(traj_file):
            mrc = mrcfile.open(traj_file, mode='r+')
        else:
            mrc = mrcfile.new(traj_file)
        mrc.set_data(imgs * -1)
        mrc.close()

        df = pandas.DataFrame(data=dict(phi=phi, theta=theta, psi=psi))
        star.write_star(ang_file, traj_file_rel, df)

    if argv:
        argv[0].emit(int((bin / params.states_per_coord) * 99))
