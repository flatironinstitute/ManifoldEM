import copy
import mrcfile
import os
import pandas

import numpy as np

from ManifoldEM import myio, util, quaternion, star, p
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''


def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    # S.M. June 2020
    offset = 0
    pathw = p.width_1D

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
            imgss = [[] for i in range(p.nClass)]
            phis = [[] for i in range(p.nClass)]
            thetas = [[] for i in range(p.nClass)]
            psis = [[] for i in range(p.nClass)]

            numNext = min(numberOfJobs, num + numberOfWorkers)

            xSel = xSelect[num:numNext]
            for x in xSel:
                offset += 1
                EL_file = p.get_EL_file(x)
                File = '{}_{}_{}'.format(EL_file, p.trajName, 1)
                data = myio.fin1(File)

                IMGT = data['IMGT']

                posPath = posPathAll[x]
                psi1Path = posPsi1All[x]

                dist_file = p.get_dist_file(x)
                data = myio.fin1(dist_file)
                q = data['q']

                q = q[:, posPath[psi1Path]]
                nS = q.shape[1]

                conOrder = nS // p.conOrderRange
                copies = conOrder
                q = q[:, copies - 1:nS - conOrder]

                IMGT = IMGT / conOrder
                IMGT = IMGT.T  # flip here IMGT is now num_images x dim^2

                tau = trajTaus[x]
                tauEq = util.hist_match(tau, tauAvg)

                for bin in range(p.nClass - pathw + 1):
                    if bin == p.nClass - pathw:
                        tauBin = ((tauEq >= (float(bin) / p.nClass)) &
                                  (tauEq <= (bin + float(pathw)) / p.nClass)).nonzero()[0]
                    else:
                        tauBin = ((tauEq >= (float(bin) / p.nClass)) &
                                  (tauEq < (bin + float(pathw)) / p.nClass)).nonzero()[0]

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

            traj_bin_file = "{}name{}_group_{}_{}".format(p.traj_file, p.trajName, num, numNext - 1)
            myio.fout1(traj_bin_file, imgss=imgss, phis=phis, thetas=thetas, psis=psis)

            print('Done saving group.')
    else:
        print('Reading previously generated trajectory data from saved projection directions...')

    # S.M. June 2020
    # loop through the nClass again and convert each list in the list to array
    for bin in range(0, p.nClass - pathw + 1):
        print('\nConcatenated bin:', bin)

        for num in range(0, numberOfJobs, numberOfWorkers):
            numNext = min(numberOfJobs, num + numberOfWorkers)

            traj_bin_file = "{}name{}_group_{}_{}".format(p.traj_file, p.trajName, num, numNext - 1)

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

        traj_file_rel = 'imgsRELION_{}_{}_of_{}.mrcs'.format(p.trajName, bin + 1, p.nClass)
        traj_file = '{}{}'.format(p.relion_dir, traj_file_rel)
        ang_file = '{}EulerAngles_{}_{}_of_{}.star'.format(p.relion_dir, p.trajName, bin + 1, p.nClass)

        if os.path.exists(traj_file):
            mrc = mrcfile.open(traj_file, mode='r+')
        else:
            mrc = mrcfile.new(traj_file)
        mrc.set_data(imgs * -1)
        mrc.close()

        df = pandas.DataFrame(data=dict(phi=phi, theta=theta, psi=psi))
        star.write_star(ang_file, traj_file_rel, df)

    if argv:
        progress7 = argv[0]
        signal = int((bin / float(p.nClass)) * 100)
        if signal == 100:
            signal = 95
        progress7.emit(signal)

    return 'ok'
