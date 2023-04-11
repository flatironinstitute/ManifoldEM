import multiprocessing
import os
import time

import numpy as np

from functools import partial
from contextlib import contextmanager
from subprocess import Popen

from ManifoldEM import p, psiAnalysisParS2, myio, ComputeEnergy1D
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def fileCheck():
    fin_PDs = []  #collect list of previously finished PDs from ELConc{}/
    for root, dirs, files in os.walk(p.EL_prog):
        for file in sorted(files):
            if not file.startswith('.'):  #ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs


def divide1(R, psiNumsAll, sensesAll):
    ll = []
    fin_PDs = fileCheck()
    for prD in R:
        dist_file = p.get_dist_file(prD)
        psi_file = p.get_psi_file(prD)
        psi2_file = p.get_psi2_file(prD)
        EL_file = p.get_EL_file(prD)
        psinums = [psiNumsAll[0, prD]]
        senses = [sensesAll[0, prD]]
        if prD not in fin_PDs:
            ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD])

    return ll


def count1(R):
    c = len(R) - len(fileCheck())
    return c


def op(*argv):
    p.load()
    #p.print()

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

    print("Processing {} projection directions.".format(len(input_data)))

    if p.ncpu == 1:  # avoids the multiprocessing package
        for i in range(len(input_data)):
            psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull, p.num_psiTrunc)
            if argv:
                offset += 1
                progress6.emit(int((offset / float(len(R))) * 99))
    else:
        with poolcontext(processes=p.ncpu) as pool:
            for _ in pool.imap_unordered(
                    partial(psiAnalysisParS2.op,
                            conOrderRange=p.conOrderRange,
                            traj_name=p.trajName,
                            isFull=isFull,
                            psiTrunc=p.num_psiTrunc),
                    input_data):
                if argv:
                    offset += 1
                    progress6.emit((offset / float(len(R))) * 99)

            pool.close()
            pool.join()

    ComputeEnergy1D.op()
    p.save()
    progress6.emit(100)


if __name__ == '__main__':
    print("Recomputing the NLSA snapshots using the found reaction coordinates...")

    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.create_dir()

    op()
