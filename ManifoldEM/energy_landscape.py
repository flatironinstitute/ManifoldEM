import multiprocessing

import numpy as np

from functools import partial

from ManifoldEM import myio, ComputeEnergy1D
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.psi_analysis import psi_analysis_single
from ManifoldEM.util import NullEmitter

"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
"""


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

    ComputeEnergy1D.op()
    params.project_level = ProjectLevel.ENERGY_LANDSCAPE
    params.save()
    progress6.emit(100)
