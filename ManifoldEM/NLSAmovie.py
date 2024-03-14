import os
import multiprocessing
import tqdm

import imageio
import numpy as np

from functools import partial

from ManifoldEM import myio
from ManifoldEM.params import p
from ManifoldEM.util import NullEmitter
from ManifoldEM.core import makeMovie
'''
% scriptPsiNLSAmovie
% Matlab Version V1.2
% Copyright(c) UWM, Ali Dashti 2016
% This script makes the NLSA movies along each reaction coordinate.
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

def _construct_input_data(N):
    ll = []
    for prD in range(N):
        image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
        if os.path.exists(image_file):
            continue
        ll.append([prD])
    return ll


def movie(input_data, psi2_file, fps):
    prD = input_data[0]
    dist_file1 = p.get_dist_file(prD)
    # Fetching NLSA outputs and making movies
    for psinum in range(p.num_psis):
        psi_file1 = psi2_file + 'prD_{}'.format(prD) + '_psi_{}'.format(psinum)
        data = myio.fin1(psi_file1)
        # make movie
        makeMovie(data['IMG1'], prD, psinum, fps)

        # write topos
        topo = data['Topo_mean'][:, 1]
        dim = int(np.sqrt(topo.shape[0]))
        image_file = '{}/topos/PrD_{}/topos_{}.png'.format(p.out_dir, prD + 1, psinum + 1)
        topo = topo.reshape(dim, dim)
        topo = (255. * (topo - topo.min()) / (topo.max() - topo.min())).astype(np.uint8)
        imageio.imwrite(image_file, topo)


    # write class avg image
    data = myio.fin1(dist_file1)
    img = data['imgAvg']
    image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
    img = (255. * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    imageio.imwrite(image_file, img)


def op(*argv):
    print("Making the 2D movies...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    input_data = _construct_input_data(p.numberofJobs)
    n_jobs = len(input_data)
    progress4 = argv[0] if use_gui_progress else NullEmitter()
    movie_local = partial(movie, psi2_file=p.psi2_file, fps=p.fps)

    if p.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            movie_local(datai)
            progress4.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(movie_local, input_data)),
                                  total=n_jobs, disable=use_gui_progress):
                progress4.emit(int(99 * i / n_jobs))

    p.save()
    progress4.emit(100)
