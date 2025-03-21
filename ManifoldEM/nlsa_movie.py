import os
import multiprocessing

import imageio
import numpy as np

from functools import partial
from typing import List, Union

from ManifoldEM import myio
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.util import NullEmitter, get_tqdm
from ManifoldEM.core import makeMovie
'''
% scriptPsinlsa_movie
% Matlab Version V1.2
% Copyright(c) UWM, Ali Dashti 2016
% This script makes the NLSA movies along each reaction coordinate.
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

def _construct_input_data(prd_list: Union[List[int], None], N):

    valid_prds = set(range(N))
    if prd_list is not None:
        requested_prds = set(prd_list)
        invalid_prds = requested_prds.difference(valid_prds)
        if invalid_prds:
            print(f"Warning: requested invalid prds: {invalid_prds}")
        valid_prds = valid_prds.intersection(requested_prds)

    jobs = []
    for prD in valid_prds:
        image_file = '{}/topos/PrD_{}/class_avg.png'.format(params.out_dir, prD + 1)
        if os.path.exists(image_file):
            continue
        jobs.append([prD])

    if jobs:
        # Sort by largest amount of work first, so the last job to run is always the shortest
        prds = data_store.get_prds()
        indices_full = prds.thresholded_image_indices
        image_counts = [len(indices_full[i]) for i in valid_prds]
        jobs, _ = zip(*sorted(zip(jobs, image_counts), key=lambda x: x[1], reverse=True))

    return jobs


def movie(input_data, fps):
    prD = input_data[0]
    dist_file1 = params.get_dist_file(prD)
    # Fetching NLSA outputs and making movies
    for psinum in range(params.num_psi):
        psi_file1 = params.get_psi2_file(prD, psinum)
        data = myio.fin1(psi_file1)
        # make movie
        makeMovie(data['IMG1'], prD, psinum, fps)

        # write topos
        topo = data['Topo_mean'][:, 1]
        dim = int(np.sqrt(topo.shape[0]))
        image_file = params.get_topos_path(prD + 1, psinum + 1)
        topo = topo.reshape(dim, dim)
        topo = (255. * (topo - topo.min()) / (topo.max() - topo.min())).astype(np.uint8)
        imageio.imwrite(image_file, topo)


    # write class avg image
    data = myio.fin1(dist_file1)
    img = data['imgAvg']
    image_file = '{}/topos/PrD_{}/class_avg.png'.format(params.out_dir, prD + 1)
    img = (255. * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    imageio.imwrite(image_file, img)


def op(prd_list: Union[List[int], None] = None, *argv):
    print("Making the 2D movies...")
    params.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    input_data = _construct_input_data(prd_list, params.prd_n_active)
    n_jobs = len(input_data)
    progress4 = argv[0] if use_gui_progress else NullEmitter()
    movie_local = partial(movie, fps=params.nlsa_fps)

    tqdm = get_tqdm()
    if params.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            movie_local(datai)
            progress4.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=params.ncpu) as pool:
            for i, _ in tqdm(enumerate(pool.imap_unordered(movie_local, input_data)),
                             total=n_jobs, disable=use_gui_progress):
                progress4.emit(int(99 * i / n_jobs))

    if not prd_list:
        params.project_level = ProjectLevel.NLSA_MOVIE

    params.save()
    progress4.emit(100)
