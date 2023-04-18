import multiprocessing

import numpy as np

from functools import partial

from ManifoldEM import p, psiAnalysisParS2
from ManifoldEM.util import NullEmitter
import tqdm
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Evan Seitz 2019 (python version)
'''


def _construct_input_data(N, rc, fin_PDs):
    ll = []
    for prD in range(N):
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        psi2_file = p.get_psi2_file(prD)
        EL_file = p.get_EL_file(prD)
        psinums = rc['psiNumsAll'][prD, :]
        senses = rc['sensesAll'][prD, :]
        psi_list = []  # list of incomplete psi values per PD
        for psi in range(len(psinums)):
            if fin_PDs[int(prD), int(psi)] == int(1):
                continue
            psi_list.append(psi)

        ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD, psi_list])
    return ll


def op(*argv):
    print("Computing the NLSA snapshots...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    psi_nums_all = np.tile(np.array(range(p.num_psis)), (p.numberofJobs, 1))  # numberofJobs x num_psis
    senses_all = np.tile(np.ones(p.num_psis), (p.numberofJobs, 1))  # numberofJobs x num_psis
    rc = {'psiNumsAll': psi_nums_all, 'sensesAll': senses_all}

    fin_PDs = np.zeros(shape=(p.numberofJobs, p.num_psis), dtype=int)
    input_data = _construct_input_data(p.numberofJobs, rc, fin_PDs)
    n_jobs = len(input_data)
    progress3 = argv[0] if use_gui_progress else NullEmitter()
    local_psi_func = partial(psiAnalysisParS2.op,
                             conOrderRange=p.conOrderRange,
                             traj_name=p.trajName,
                             isFull=0,
                             psiTrunc=p.num_psiTrunc)

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
