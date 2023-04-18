import os
import multiprocessing
import numpy as np

from functools import partial

from ManifoldEM import myio, p, getDistanceCTF_local_Conj9combinedS2
from ManifoldEM.util import NullEmitter
import tqdm
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def _split_input(CG, q, df, N):
    ll = []
    for prD in range(N):
        ind = CG[prD]
        q1 = q[:, ind]
        df1 = df[ind]
        dist_file = p.get_dist_file(prD)
        ll.append([ind, q1, df1, dist_file, prD])
    return ll


def op(*argv):
    print("Computing the distances...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv)

    data = myio.fin1(p.tess_file)
    (CG, df, q, sh) = data['CG'], data['df'], data['q'], data['sh']

    filterPar = dict(type='Butter', Qc=0.5, N=8)
    options = dict(verbose=False,
                   avgOnly=False,
                   visual=False,
                   parallel=False,
                   relion_data=p.relion_data,
                   thres=p.PDsizeThH)

    # SPIDER only: compute the nPix = sqrt(len(bin file)/(4*num_part))
    if p.relion_data is False:
        p.nPix = int(np.sqrt(os.path.getsize(p.img_stack_file) / (4 * p.num_part)))
        p.save()

    input_data = _split_input(CG, q, df, p.numberofJobs)
    n_jobs = len(input_data)
    local_distance_func = partial(getDistanceCTF_local_Conj9combinedS2.op,
                                  filterPar=filterPar,
                                  imgFileName=p.img_stack_file,
                                  sh=sh,
                                  nStot=len(df),
                                  options=options)

    if use_gui_progress:
        progress1 = argv[0]
    else:
        progress1 = NullEmitter()

    print(f"Processing {len(input_data)} projection directions for distance calculation")
    if p.ncpu == 1 or options['parallel'] is True:
        for i, datai in tqdm.tqdm(enumerate(input_data),
                                  total=n_jobs, disable=use_gui_progress):
            local_distance_func(datai)
            progress1.emit(int((i / len(input_data)) * 99))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(
                    enumerate(pool.imap_unordered(local_distance_func, input_data)),
                    total=n_jobs, disable=use_gui_progress):
                progress1.emit(int((i / p.numberofJobs) * 99))

    p.save()
    progress1.emit(100)
