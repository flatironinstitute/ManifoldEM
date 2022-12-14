import sys
import os
import time
import multiprocessing

from contextlib import contextmanager
from functools import partial
from subprocess import Popen

from ManifoldEM import manifoldAnalysis_mpi, manifoldTrimmingAuto, myio, p, set_params
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.close()


def fileCheck():
    fin_PDs = []  # collect list of previously finished PDs from diff_maps/progress/
    for root, dirs, files in os.walk(p.psi_prog):
        for file in sorted(files):
            if not file.startswith('.'):  # ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs


def count(N):
    c = N - len(fileCheck())
    return c


def divide(N):
    ll = []
    fin_PDs = fileCheck()
    for prD in range(N):
        dist_file = '{}prD_{}'.format(p.dist_file, prD)
        psi_file = '{}prD_{}'.format(p.psi_file, prD)
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, prD + 1)
        if prD not in fin_PDs:
            ll.append([dist_file, psi_file, eig_file, prD])
    return ll


def op(*argv):
    time.sleep(5)
    set_params.op(1)
    #set_params.op(-1)

    multiprocessing.set_start_method('fork', force=True)

    if p.machinefile:
        print('using MPI with {} processes'.format(p.ncpu))
        Popen([
            "mpirun", "-n",
            str(p.ncpu), "-machinefile",
            str(p.machinefile), "python", "modules/manifoldAnalysis_mpi.py",
            str(p.proj_name)
        ],
              close_fds=True)
        for i in range(p.numberofJobs):
            subdir = p.out_dir + '/topos/PrD_{}'.format(i + 1)
            os.makedirs(subdir, exist_ok=True)
        if argv:
            progress2 = argv[0]
            offset = 0
            while offset < p.numberofJobs:
                offset = p.numberofJobs - count(p.numberofJobs)
                progress2.emit(int((offset / float(p.numberofJobs)) * 100))
                time.sleep(5)

    else:
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
            progress2.emit(int((offset / float(p.numberofJobs)) * 100))

        print("Processing {} projection directions.".format(len(input_data)))
        for i in range(p.numberofJobs):
            subdir = p.out_dir + '/topos/PrD_{}'.format(i + 1)
            os.makedirs(subdir, exist_ok=True)

        if p.ncpu == 1:  #avoids the multiprocessing package
            for i in range(len(input_data)):
                manifoldTrimmingAuto.op(input_data[i], posPath, p.tune, p.rad, visual, doSave)
                if argv:
                    offset += 1
                    progress2.emit(int((offset / float(p.numberofJobs)) * 100))
        else:
            with poolcontext(processes=p.ncpu, maxtasksperchild=1) as pool:
                for i, _ in enumerate(
                        pool.imap_unordered(
                            partial(manifoldTrimmingAuto.op,
                                    posPath=posPath,
                                    tune=p.tune,
                                    rad=p.rad,
                                    visual=visual,
                                    doSave=doSave), input_data), 1):
                    if argv:
                        offset += 1
                        progress2.emit(int((offset / float(p.numberofJobs)) * 100))
                    time.sleep(0.05)
                pool.close()
                pool.join()

    set_params.op(0)


if __name__ == '__main__':
    p.init()
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.create_dir()
    op()
