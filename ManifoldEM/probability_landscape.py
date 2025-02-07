import multiprocessing
import os

import numpy as np

from functools import partial

from ManifoldEM import myio
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.psi_analysis import psi_analysis_single
from ManifoldEM.util import NullEmitter, debug_print

"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
"""


def probability_landscape_local():
    print("Computing the probability landscape...")
    params.load()

    data = myio.fin1(params.CC_file)
    psiNumsAll = data['psinums']

    xSelect = np.arange(params.prd_n_active)
    unused_prds = set(np.nonzero(psiNumsAll[0, :] == -1)[0])  # unassigned states
    unused_prds = unused_prds.union(data_store.get_prds().trash_ids)

    xSelect = np.delete(xSelect, list(unused_prds))

    if not len(xSelect):
        debug_print("No assigned states: Unable to compute state profile")


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
        h, _ = np.histogram(tau, params.states_per_coord)
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

    return hUn


def divide1(R, psiNumsAll, sensesAll):
    ll = []
    prds = data_store.get_prds()
    for prD in R:
        dist_file = params.get_dist_file(prD)
        psi_file = params.get_psi_file(prD)
        EL_file = params.get_EL_file(prD)
        psinums = [psiNumsAll[0, prD]]
        senses = [sensesAll[0, prD]]
        defocus = prds.get_defocus_by_prd(prD)
        ll.append([dist_file, psi_file, EL_file, psinums, senses, prD, defocus])

    return ll


def op(*argv):
    params.load()

    multiprocessing.set_start_method("fork", force=True)

    R = np.array(range(params.prd_n_active))
    R = np.delete(R, list(data_store.get_prds().trash_ids))

    print("Recomputing the NLSA snapshots using the found reaction coordinates...")
    data = myio.fin1(params.CC_file)
    psiNumsAll = data["psinums"]
    sensesAll = data["senses"]
    isFull = 1
    input_data = divide1(R, psiNumsAll, sensesAll)
    if argv:
        progress6 = argv[0]
        offset = len(R) - len(input_data)
        progress6.emit(int((offset / float(len(R))) * 99))
    else:
        progress6 = NullEmitter()
        offset = 0

    print(f"Processing {len(input_data)} projection directions.")

    local_func = partial(
        psi_analysis_single,
        con_order_range=params.con_order_range,
        traj_name=params.traj_name,
        is_full=isFull,
        psi_trunc=params.num_psi_truncated,
    )

    if params.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in enumerate(input_data):
            local_func(datai)
            progress6.emit(int(((offset + i) / len(R)) * 99))
    else:
        with multiprocessing.Pool(processes=params.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(local_func, input_data)):
                progress6.emit(((offset + i) / len(R)) * 99)

    probability_landscape_local()
    params.project_level = ProjectLevel.PROBABILITY_LANDSCAPE
    params.save()

    progress6.emit(100)
