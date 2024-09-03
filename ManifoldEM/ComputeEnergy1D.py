import os

import numpy as np

from ManifoldEM import myio
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params
from ManifoldEM.util import debug_print
''' %Version V 1.2
    % Copyright (c) UWM, Ali Dashti 2016 (matlab version)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %This script prepares the image stacks and orientations for 3D reconstruction.
    Copyright (c) Columbia Univ Hstau Liao 2018 (python version)
    Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''


def op(*argv):
    print("Computing the energy landscape...")
    params.load()

    data = myio.fin1(params.CC_file)
    psiNumsAll = data['psinums']

    xSelect = np.arange(params.prd_n_active)
    unused_prds = set(np.nonzero(psiNumsAll[0, :] == -1)[0])  # unassigned states
    unused_prds = unused_prds.union(data_store.get_prds().trash_ids)

    xSelect = np.delete(xSelect, list(unused_prds))

    if not len(xSelect):
        debug_print("No assigned states: Unable to compute energy profile")


    # getFromFileS2
    xLost = []
    trajTaus = [None] * params.prd_n_active
    posPathAll = [None] * params.prd_n_active
    posPsi1All = [None] * params.prd_n_active

    for x in xSelect:
        File = params.get_EL_file(x)

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
    hUn = np.zeros((1, params.states_per_coord)).flatten()
    tauAvg = np.array([])

    for x in xSelect:
        tau = trajTaus[x]
        tau = tau.flatten()

        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        h, ctrs = np.histogram(tau, params.states_per_coord)
        hUn = hUn + h
        tauAvg = np.concatenate((tauAvg, tau.flatten()))

    # Section III
    traj_file = "{}name{}.pkl".format(params.traj_file, params.traj_name)
    myio.fout1(traj_file, hUn=hUn)

    # added June 2020, S.M.
    traj_file_vars = f"{params.traj_file}name{params.traj_name}_vars.pkl"
    myio.fout1(traj_file_vars, trajTaus=trajTaus, posPsi1All=posPsi1All,
               posPathAll=posPathAll, xSelect=xSelect, tauAvg=tauAvg)

    hUn.astype('int').tofile(params.OM_file)

    #################
    # compute energy:
    T = params.temperature  # Celsius, may need to be user-defined
    kB = 0.0019872041  # Boltzmann constant kcal / Mol / K
    rho = np.fmax(hUn, 1)
    kT = kB * (T + 273.15)  # Kelvin
    E = -kT * np.log(rho)
    E = E - np.amin(E)  # shift so that lowest energy is zero
    E.astype('float').tofile(params.OM1_file)

    params.save()

    if argv:
        progress7 = argv[0]
        progress7.emit(100)

    return hUn
