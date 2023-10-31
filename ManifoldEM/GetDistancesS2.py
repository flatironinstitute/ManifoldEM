import multiprocessing
import tqdm

from functools import partial

from ManifoldEM import getDistanceCTF_local_Conj9combinedS2
from ManifoldEM.data_store import data_store
from ManifoldEM.params import p
from ManifoldEM.util import NullEmitter

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


def _construct_input_data(thresholded_indices, quats_full, defocus):
    ll = []
    for prD in range(len(thresholded_indices)):
        ind = thresholded_indices[prD]
        ll.append({'indices': ind,
                   'quats': quats_full[:, ind],
                   'defocus': defocus[ind],
                   'dist_file': p.get_dist_file(prD)})

    return ll


def op(*argv):
    print("Computing the distances...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    prds = data_store.get_prds()

    filterPar = dict(type='Butter', Qc=0.5, N=8)

    input_data = _construct_input_data(prds.thresholded_image_indices, prds.quats_full, prds.defocus)
    n_jobs = len(input_data)
    local_distance_func = partial(getDistanceCTF_local_Conj9combinedS2.op,
                                  filter_par=filterPar,
                                  img_file_name=p.img_stack_file,
                                  image_offsets=prds.microscope_origin,
                                  n_particles_tot=len(prds.defocus),
                                  avg_only=False,
                                  relion_data=p.relion_data
                                  )

    progress1 = argv[0] if use_gui_progress else NullEmitter()

    if p.ncpu == 1:
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            local_distance_func(datai)
            progress1.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(
                    enumerate(pool.imap_unordered(local_distance_func, input_data)),
                    total=n_jobs, disable=use_gui_progress):
                progress1.emit(int(99 * i / n_jobs))

    p.save()
    progress1.emit(100)
