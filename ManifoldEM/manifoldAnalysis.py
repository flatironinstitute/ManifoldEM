import os
import multiprocessing

from functools import partial
from typing import List, Union

from ManifoldEM import manifoldTrimmingAuto
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


def _construct_input_data(prd_list, N):
    ll = []

    valid_prds = set(range(N))
    if prd_list is not None:
        requested_prds = set(prd_list)
        invalid_prds = requested_prds.difference(valid_prds)
        if invalid_prds:
            print(f"Warning: requested invalid prds: {invalid_prds}")
        valid_prds = valid_prds.intersection(requested_prds)

    for prD in valid_prds:
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, prD + 1)
        ll.append([dist_file, psi_file, eig_file, prD])

    return ll


def op(prd_list: Union[List[int], None], *argv):
    print("Computing the eigenfunctions...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    # Finding and trimming manifold from particles
    input_data = _construct_input_data(prd_list, p.numberofJobs)
    n_jobs = len(input_data)
    progress2 = argv[0] if use_gui_progress else NullEmitter()

    for i in range(n_jobs):
        subdir = os.path.join(p.out_dir, 'topos', f'PrD_{i+1}')
        os.makedirs(subdir, exist_ok=True)

    local_trim_func = partial(manifoldTrimmingAuto.op,
                              posPath=0,
                              tune=p.tune,
                              rad=p.rad,
                              visual=False,
                              doSave=dict(outputFile='', Is=True))

    if p.ncpu == 1:
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            local_trim_func(datai)
            progress2.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(local_trim_func, input_data)),
                                  total=n_jobs,
                                  disable=use_gui_progress):
                progress2.emit(int(99 * i / n_jobs))

    p.save()
    progress2.emit(100)
