import os
import multiprocessing

from functools import partial

from ManifoldEM import manifoldTrimmingAuto, p
from ManifoldEM.util import NullEmitter
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def divide(N):
    ll = []
    for prD in range(N):
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, prD + 1)
        ll.append([dist_file, psi_file, eig_file, prD])
    return ll


def op(*argv):
    p.load()

    multiprocessing.set_start_method('fork', force=True)

    print("Computing the eigenfunctions...")
    doSave = dict(outputFile='', Is=True)
    # INPUT Parameters
    visual = False
    posPath = 0
    # Finding and trimming manifold from particles
    input_data = divide(p.numberofJobs)
    if argv:
        progress2 = argv[0]
        offset = p.numberofJobs - len(input_data)
        progress2.emit(int((offset / float(p.numberofJobs)) * 99))
    else:
        progress2 = NullEmitter()
        offset = 0

    print(f"Processing {len(input_data)} projection directions.")
    for i in range(p.numberofJobs):
        subdir = p.out_dir + '/topos/PrD_{}'.format(i + 1)
        os.makedirs(subdir, exist_ok=True)

    if p.ncpu == 1:
        for i, datai in enumerate(input_data):
            manifoldTrimmingAuto.op(datai, posPath, p.tune, p.rad, visual, doSave)
            progress2.emit(int((offset + i / p.numberofJobs) * 99))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(
                    partial(manifoldTrimmingAuto.op,
                            posPath=posPath,
                            tune=p.tune,
                            rad=p.rad,
                            visual=visual,
                            doSave=doSave),
                    input_data)):
                progress2.emit(int(((offset + i) / p.numberofJobs) * 99))

    p.save()
    progress2.emit(100)
