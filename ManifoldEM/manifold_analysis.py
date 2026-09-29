import os
import multiprocessing

from functools import partial
from typing import List, Union

from ManifoldEM import manifoldTrimmingAuto
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.util import get_tqdm
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
        dist_file = params.get_dist_file(prD)
        psi_file = params.get_psi_file(prD)
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(params.out_dir, prD + 1)
        ll.append([dist_file, psi_file, eig_file, prD])

    return ll


def op(prd_list: Union[List[int], None] = None):
    """
    Orchestrates the processing of multiple datasets for manifold learning and trimming,
    utilizing multiprocessing for parallel execution.

    Parameters
    ----------
    prd_list : Union[List[int], None]
        If `None`, process all available prds and update project level.
        Otherwise process only those in the list without updating the project level.

    Notes:
    - The function begins by loading configuration parameters using `p.load()`.
    - It sets the multiprocessing start method to 'fork' to optimize for certain environments.
    - It constructs input data configurations for each dataset using `_construct_input_data`.
    - Depending on the number of CPUs specified in the configuration (`p.ncpu`), it either
      processes the datasets sequentially or in parallel using a multiprocessing pool.
    - Configuration parameters are saved after processing using `p.save()`.
    """

    print("Computing the eigenfunctions...")
    params.load()
    multiprocessing.set_start_method('fork', force=True)

    # Finding and trimming manifold from particles
    input_data = _construct_input_data(prd_list, params.prd_n_active)
    n_jobs = len(input_data)

    for i in range(n_jobs):
        subdir = os.path.join(params.out_dir, 'topos', f'PrD_{i+1}')
        os.makedirs(subdir, exist_ok=True)

    local_trim_func = partial(manifoldTrimmingAuto.op,
                              posPath=0,
                              tune=params.nlsa_tune,
                              rad=params.rad,
                              visual=False,
                              doSave=dict(outputFile='', Is=True))

    tqdm = get_tqdm()
    if params.ncpu == 1:
        for datai in tqdm(input_data, total=n_jobs):
            local_trim_func(datai)
    else:
        with multiprocessing.Pool(processes=params.ncpu) as pool:
            for _ in tqdm(pool.imap_unordered(local_trim_func, input_data), total=n_jobs):
                pass

    if prd_list is None:
        params.project_level = ProjectLevel.MANIFOLD_ANALYSIS

    params.save()
