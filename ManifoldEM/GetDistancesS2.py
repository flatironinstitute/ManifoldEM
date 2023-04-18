import os
import multiprocessing
import numpy as np

from functools import partial

from ManifoldEM import myio, p, getDistanceCTF_local_Conj9combinedS2, Data
from ManifoldEM.util import NullEmitter
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def divide(CG, q, df, N):
    ll = []
    for prD in range(N):
        ind = CG[prD]
        q1 = q[:, ind]
        df1 = df[ind]
        dist_file = p.get_dist_file(prD)
        ll.append([ind, q1, df1, dist_file, prD])
    return ll


def op(*argv):
    p.load()

    multiprocessing.set_start_method('fork', force=True)

    data = myio.fin1(p.tess_file)
    CG = data['CG']

    print("Computing the distances...")
    df = data['df']
    q = data['q']
    sh = data['sh']
    p.load()
    size = len(df)

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
        p.save()  #send new GUI data to user parameters file

    input_data = divide(CG, q, df, p.numberofJobs)
    if argv:
        progress1 = argv[0]
        offset = p.numberofJobs - len(input_data)
        progress1.emit(int((offset / float(p.numberofJobs)) * 100))
    else:
        progress1 = NullEmitter()
        offset = 0

    print(f"Processing {len(input_data)} projection directions.")

    if p.ncpu == 1 or options['parallel'] is True:
        for i, datai in enumerate(input_data):
            getDistanceCTF_local_Conj9combinedS2.op(datai, filterPar, p.img_stack_file, sh, size, options)
            progress1.emit(int(((offset + i) / float(p.numberofJobs)) * 99))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(
                    partial(getDistanceCTF_local_Conj9combinedS2.op,
                            filterPar=filterPar,
                            imgFileName=p.img_stack_file,
                            sh=sh,
                            nStot=size,
                            options=options), input_data)):
                progress1.emit(int(((offset + i) / p.numberofJobs) * 99))

    p.save()
    progress1.emit(100)
