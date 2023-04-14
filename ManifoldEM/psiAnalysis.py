import multiprocessing

import numpy as np

from functools import partial
from contextlib import contextmanager

from ManifoldEM import p, psiAnalysisParS2
from ManifoldEM.util import NullEmitter

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Evan Seitz 2019 (python version)
'''


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def divid(N, rc, fin_PDs):
    ll = []
    for prD in range(N):
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        psi2_file = p.get_psi2_file(prD)
        EL_file = p.get_EL_file(prD)
        psinums = rc['psiNumsAll'][prD, :]
        senses = rc['sensesAll'][prD, :]
        psi_list = []  #list of incomplete psi values per PD
        for psi in range(len(psinums)):
            if fin_PDs[int(prD), int(psi)] == int(1):
                continue
            else:
                psi_list.append(psi)
        ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD, psi_list])
    return ll


def op(*argv):
    p.load()

    multiprocessing.set_start_method('fork', force=True)

    psiNumsAll = np.tile(np.array(range(p.num_psis)), (p.numberofJobs, 1))  # numberofJobs x num_psis
    sensesAll = np.tile(np.ones(p.num_psis), (p.numberofJobs, 1))  # numberofJobs x num_psis
    rc = {'psiNumsAll': psiNumsAll, 'sensesAll': sensesAll}

    print("Computing the NLSA snapshots...")
    isFull = 0
    fin_PDs = np.zeros(shape=(p.numberofJobs, p.num_psis), dtype=int)
    input_data = divid(p.numberofJobs, rc, fin_PDs)

    if argv:
        progress3 = argv[0]
    else:
        progress3 = NullEmitter()

    print(f"Processing {len(input_data)} projection directions.")

    if p.ncpu == 1:  # avoids the multiprocessing package
        for i in range(len(input_data)):
            if argv:  #for p.ncpu=1, progress3 update happens inside psiAnalysisParS2;
                # however, same signal can't be sent if multiprocessing
                psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull, p.num_psiTrunc, argv[0])
            else:  #for non-GUI
                psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull, p.num_psiTrunc)
            progress3.emit(int(99 * (i / p.numberofJobs)))
    else:
        with poolcontext(processes=p.ncpu) as pool:
            for i, _ in enumerate(
                    pool.imap_unordered(
                        partial(psiAnalysisParS2.op,
                                conOrderRange=p.conOrderRange,
                                traj_name=p.trajName,
                                isFull=isFull,
                                psiTrunc=p.num_psiTrunc),
                        input_data)):
                if argv:
                    argv[0].emit(int(99 * (i / p.numberofJobs)))

            pool.close()
            pool.join()

    p.save()
    progress3.emit(100)
