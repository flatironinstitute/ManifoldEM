import os
import time
import multiprocessing
import numpy as np

from functools import partial
from contextlib import contextmanager
from subprocess import Popen

from ManifoldEM import myio, p, getDistanceCTF_local_Conj9combinedS2, Data
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
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
    fin_PDs = []  #collect list of previously finished PDs from distances/progress/
    for root, dirs, files in os.walk(p.dist_prog):
        for file in sorted(files):
            if not file.startswith('.'):  #ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs


def divide(CG, q, df, N):
    ll = []
    fin_PDs = fileCheck()
    for prD in range(N):
        ind = CG[prD]
        q1 = q[:, ind]
        df1 = df[ind]
        dist_file = p.get_dist_file(prD)
        if prD not in fin_PDs:
            ll.append([ind, q1, df1, dist_file, prD])
    return ll


def count(N):
    c = N - len(fileCheck())
    return c


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

    sigmaH = 0

    # SPIDER only: compute the nPix = sqrt(len(bin file)/(4*num_part))
    if p.relion_data is False:
        p.nPix = int(np.sqrt(os.path.getsize(p.img_stack_file) / (4 * p.num_part)))
        p.save()  #send new GUI data to user parameters file

    input_data = divide(CG, q, df, p.numberofJobs)
    if argv:
        progress1 = argv[0]
        offset = p.numberofJobs - len(input_data)
        progress1.emit(int((offset / float(p.numberofJobs)) * 100))

    print("Processing {} projection directions.".format(len(input_data)))

    if p.ncpu == 1 or options['parallel'] == True:  # avoids the multiprocessing package
        for i in range(len(input_data)):
            getDistanceCTF_local_Conj9combinedS2.op(input_data[i], filterPar, p.img_stack_file, sh, size, options)
            if argv:
                offset += 1
                progress1.emit(int((offset / float(p.numberofJobs)) * 99))
    else:
        with poolcontext(processes=p.ncpu) as pool:
            for _ in pool.imap_unordered(
                    partial(getDistanceCTF_local_Conj9combinedS2.op,
                            filterPar=filterPar,
                            imgFileName=p.img_stack_file,
                            sh=sh,
                            nStot=size,
                            options=options), input_data):
                if argv:
                    offset += 1
                    progress1.emit(int((offset / float(p.numberofJobs)) * 99))

            pool.close()
            pool.join()

    p.save()
    progress1.emit(100)


if __name__ == '__main__':
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.align_param_file = os.path.join(p.user_dir, 'run_it300_data.star')
    p.img_stack_file = os.path.join(p.user_dir, '2_toy42_stack.mrcs')
    p.create_dir()
    Data.op(p.align_param_file)
    op()
