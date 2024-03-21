import gc
import os
import sys

import numpy as np

from ManifoldEM import writeRelionS2, myio
from ManifoldEM.data_store import data_store
from ManifoldEM.params import p, ProjectLevel

def op(*argv):
    """This script prepares the image stacks and orientations for 3D reconstruction."""
    # Copyright (c) UWM, Ali Dashti 2016 (matlab version)
    # Copyright (c) Columbia Univ Hstau Liao 2018 (python version)
    # Copyright (c) Columbia University Suvrajit Maji 2020 (python version)

    p.load()
    print("Writing output files...")

    psiNumsAll = myio.fin1(p.CC_file)['psinums']

    # read reaction coordinates
    a = set(np.nonzero(psiNumsAll[0, :] == -1)[0])  #unassigned states, python
    a = list(a.union(data_store.get_prds().trash_ids))
    xSelect = np.delete(np.arange(p.prd_n_active), a)

    # getFromFileS2
    xLost = []
    trajTaus = [None] * p.prd_n_active
    posPathAll = [None] * p.prd_n_active
    posPsi1All = [None] * p.prd_n_active

    for x in xSelect:
        EL_file = p.get_EL_file(x)
        File = '{}_{}_{}'.format(EL_file, p.traj_name, 1)
        if os.path.exists(File):
            data = myio.fin1(File)
            trajTaus[x] = data['tau']
            posPathAll[x] = data['posPath']
            posPsi1All[x] = data['PosPsi1']
        else:
            xLost.append(x)
            continue

    xSelect = list(set(xSelect) - set(xLost))

    # Section II
    tauAvg = np.array([])
    for x in xSelect:
        tau = trajTaus[x]
        tau = tau.flatten()
        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        tauAvg = np.concatenate((tauAvg, tau.flatten()))

    # added June 2020, S.M.
    traj_file2 = "{}name{}_vars".format(p.traj_file, p.traj_name)
    myio.fout1(traj_file2, trajTaus=trajTaus, posPsi1All=posPsi1All, posPathAll=posPathAll,
               xSelect=xSelect, tauAvg=tauAvg)

    # Section III
    if argv:
        writeRelionS2.op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, argv[0])
    else:
        writeRelionS2.op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg)

    p.project_level = ProjectLevel.TRAJECTORY
    p.save()

    if argv:
        progress7 = argv[0]
        progress7.emit(100)

    print('finished manifold embedding!')
