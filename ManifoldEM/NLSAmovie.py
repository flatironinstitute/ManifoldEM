import os
import multiprocessing
import tqdm

import matplotlib.pyplot as plt
import numpy as np

from functools import partial

from ManifoldEM import myio
from ManifoldEM.params import p
from ManifoldEM.util import NullEmitter
from ManifoldEM.core import makeMovie
from ManifoldEM.data_store import data_store
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
        ll.append(prD)
    return ll


def movie(prD, psi2_file, fps):
    # Fetching NLSA outputs and making movies
    IMG1All = []
    Topo_mean = []
    for psinum in range(p.num_psis):
        psi_file1 = psi2_file + 'prD_{}'.format(prD) + '_psi_{}'.format(psinum)
        data = myio.fin1(psi_file1)
        IMG1All.append(data['IMG1'])
        Topo_mean.append(data['Topo_mean'])
        # make movie
        makeMovie(IMG1All[psinum], prD, psinum, fps)

        ######################
        # write topos
        # TODO: This shouldn't require imshow. We can almost certainly just write the images directly
        topo = Topo_mean[psinum]
        dim = int(np.sqrt(topo.shape[0]))

        fig2 = plt.figure(frameon=False)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.axis('off')
        ax2.set_title('')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.imshow(topo[:, 1].reshape(dim, dim), cmap=plt.get_cmap('gray'))
        image_file = '{}/topos/PrD_{}/topos_{}.png'.format(p.out_dir, prD + 1, psinum + 1)
        fig2.savefig(image_file, bbox_inches='tight', dpi=100, pad_inches=-0.1)
        ax2.clear()
        fig2.clf()
        plt.close(fig2)

    # write class avg image
    avg = data_store.get_distances().img_avg(prD)
    fig3 = plt.figure(frameon=False)
    ax3 = fig3.add_axes([0, 0, 1, 1])
    ax3.axis('off')
    ax3.set_title('')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.imshow(avg, cmap=plt.get_cmap('gray'))
    image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
    fig3.savefig(image_file, bbox_inches='tight', dpi=100, pad_inches=-0.1)
    ax3.clear()
    fig3.clf()
    plt.close(fig3)


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
                                  total=n_jobs,
                                  disable=use_gui_progress):
                progress4.emit(int(99 * i / n_jobs))

    p.save()
    progress4.emit(100)
