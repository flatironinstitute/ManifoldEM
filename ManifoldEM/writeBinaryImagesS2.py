import h5py
import os

import numpy as np
import matplotlib.pyplot as plt

from ManifoldEM import myio, util
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)    
'''


def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    import p
    i = 0
    for x in xSelect:
        i += 1
        EL_file = '{}prD_{}'.format(p.EL_file, x)
        File = '{}_{}_{}'.format(EL_file, p.trajName, 1)
        data = myio.fin1(File)

        IMGT = data['IMGT']

        posPath = posPathAll[x]
        psi1Path = posPsi1All[x]

        dist_file = '{}prD_{}'.format(p.dist_file, x)
        data = myio.fin1(dist_file)
        q = data['q']

        q = q[:, posPath[psi1Path]]  # python
        nS = q.shape[1]

        conOrder = np.floor(float(nS) / p.conOrderRange).astype(int)
        copies = conOrder
        q = q[:, copies - 1:nS - conOrder]

        IMGT = IMGT / conOrder
        IMGT = IMGT.T  #flip here IMGT is now num_images x dim^2

        tau = trajTaus[x]
        tauEq = util.hist_match(tau, tauAvg)
        pathw = p.width_1D
        IMG1 = np.zeros((p.nClass, IMGT.shape[1]))
        for bin in range(p.nClass - pathw + 1):
            if bin == p.nClass - pathw:
                tauBin = ((tauEq >= (bin / float(p.nClass))) & (tauEq <= (bin + pathw) / p.nClass)).nonzero()[0]
            else:
                tauBin = ((tauEq >= (bin / float(p.nClass))) & (tauEq < (bin + pathw) / p.nClass)).nonzero()[0]

            if len(tauBin) == 0:
                continue
            else:
                f1 = '{}NLSAImageTraj{}_{}_of_{}.dat'.format(p.bin_dir, p.trajName, bin + 1, p.nClass)
                f2 = '{}TauValsTraj{}_{}_of_{}.dat'.format(p.bin_dir, p.trajName, bin + 1, p.nClass)
                f3 = '{}ProjDirTraj{}_{}_of_{}.dat'.format(p.bin_dir, p.trajName, bin + 1, p.nClass)
                f4 = '{}quatsTraj{}_{}_of_{}.dat'.format(p.bin_dir, p.trajName, bin + 1, p.nClass)

                ar1 = IMGT[tauBin, :]

                with open(f1, "ab") as f:  # or choose 'wb' mode
                    ar1.astype('float').tofile(f)
                ar2 = tauEq[tauBin]
                with open(f2, "ab") as f:
                    ar2.astype('float').tofile(f)
                ar3 = x * np.ones((1, len(tauBin)))
                with open(f3, "ab") as f:
                    ar3.astype('int').tofile(f)
                ar4 = q[:, tauBin]
                ar4 = ar4.flatten('F')
                with open(f4, "ab") as f:
                    ar4.astype('float64').tofile(f)

                IMG1[bin, :] = np.mean(IMGT[tauBin, :], axis=0)

        bin_file = '{}PD_{}_Traj{}'.format(p.bin_dir, x, p.trajName)
        myio.fout1(bin_file, ['IMG1'], [IMG1])

        if argv:
            progress7 = argv[0]
            signal = int((i / float(len(xSelect))) * 50)
            progress7.emit(signal)

    res = 'ok'
    return res
