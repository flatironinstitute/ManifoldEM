import os
import multiprocessing

from functools import partial

from ManifoldEM import manifoldTrimmingAuto
from ManifoldEM.data_store import data_store
from ManifoldEM.params import p
from ManifoldEM.util import NullEmitter

import tqdm
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def _construct_input_data(N):
    ll = []
    for prD in range(N):
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, prD + 1)
        ll.append([eig_file, prD])
    return ll


def op(*argv):
    print("Computing the eigenfunctions...")
    p.load()
    diff_map_store = data_store.get_diff_maps()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    # Finding and trimming manifold from particles
    input_data = _construct_input_data(p.numberofJobs)
    n_jobs = len(input_data)
    progress = argv[0] if use_gui_progress else NullEmitter()

    for i in range(n_jobs):
        subdir = os.path.join(p.out_dir, 'topos', f'PrD_{i+1}')
        os.makedirs(subdir, exist_ok=True)

    local_trim_func = partial(manifoldTrimmingAuto.op, posPath=0, tune=p.tune, rad=p.rad)

    if p.ncpu == 1:
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            (prd, res) = local_trim_func(datai)
            diff_map_store.update(prd, res)
            progress.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            jobs = pool.imap_unordered(local_trim_func, input_data)

            for i, (prd, res) in enumerate(tqdm.tqdm(jobs, total=n_jobs, disable=use_gui_progress)):
                diff_map_store.update(prd, res)
                progress.emit(int(99 * i / n_jobs))

    p.save()
    progress.emit(100)
