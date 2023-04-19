import multiprocessing

import numpy as np

from functools import partial

from ManifoldEM import p, myio, ComputeEnergy1D
from ManifoldEM.psiAnalysis import psi_analysis_single
from ManifoldEM.util import NullEmitter
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def divide1(R, psiNumsAll, sensesAll):
    ll = []
    for prD in R:
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        psi2_file = p.get_psi2_file(prD)
        EL_file = p.get_EL_file(prD)
        psinums = [psiNumsAll[0, prD]]
        senses = [sensesAll[0, prD]]
        ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD])

    return ll


def op(*argv):
    p.load()

    multiprocessing.set_start_method('fork', force=True)

    R = np.array(range(p.numberofJobs))
    R = np.delete(R, np.nonzero(p.get_trash_list())[0])

    print("Recomputing the NLSA snapshots using the found reaction coordinates...")
    data = myio.fin1(p.CC_file)
    psiNumsAll = data['psinums']
    sensesAll = data['senses']
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

    local_func = partial(psi_analysis_single,
                         con_order_range=p.conOrderRange,
                         traj_name=p.trajName,
                         is_full=isFull,
                         psi_trunc=p.num_psiTrunc)

    if p.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in enumerate(input_data):
            local_func(datai)
            progress6.emit(int(((offset + i) / len(R)) * 99))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(local_func, input_data)):
                progress6.emit(((offset + i) / len(R)) * 99)

    ComputeEnergy1D.op()
    p.save()
    progress6.emit(100)
