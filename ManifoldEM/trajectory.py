import h5pickle
import mrcfile
import multiprocessing
from nptyping import Float, NDArray, Shape, Int
import numpy as np
import os

from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, cast

from ManifoldEM.data_store import data_store
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.star import write_star
from ManifoldEM import quaternion
from ManifoldEM.util import get_tqdm, NullEmitter, hist_match
from ManifoldEM.prd_analysis import run_nlsa_second_pass


def extract_traj_data(
    tau: NDArray[Shape["Any"], Float],
    tau_avg: NDArray[Shape["Any, Any"], Float],
    psi_sorted_quats: NDArray[Shape["Any, Any"], Float],
    pathw: int,
    states_per_coord: int,
    con_order_range: int,
):
    nS = psi_sorted_quats.shape[1]
    con_order = nS // con_order_range
    q = psi_sorted_quats[:, con_order - 1 : nS - con_order]
    tau_eq = hist_match(tau, tau_avg)

    taubins = [np.empty(0) for _ in range(states_per_coord)]
    phis = [np.empty(0) for _ in range(states_per_coord)]
    thetas = [np.empty(0) for _ in range(states_per_coord)]
    psis = [np.empty(0) for _ in range(states_per_coord)]
    for i_bin in range(states_per_coord - pathw + 1):
        if i_bin == states_per_coord - pathw:
            tau_bin = (
                (tau_eq >= (i_bin / states_per_coord))
                & (tau_eq <= (i_bin + pathw) / states_per_coord)
            ).nonzero()[0]
        else:
            tau_bin = (
                (tau_eq >= (i_bin / states_per_coord))
                & (tau_eq < (i_bin + pathw) / states_per_coord)
            ).nonzero()[0]

        if not len(tau_bin):
            continue

        qs = q[:, tau_bin]
        n_tau = len(tau_bin)
        PDs = quaternion.calc_avg_pd(qs, n_tau)
        phi = np.empty(n_tau)
        theta = np.empty(n_tau)
        psi = np.empty(n_tau)

        for i in range(n_tau):
            phi[i], theta[i], psi[i] = quaternion.psi_ang(PDs[:, i])

        taubins[i_bin] = tau_bin
        phis[i_bin] = phi
        thetas[i_bin] = theta
        psis[i_bin] = psi

    return taubins, phis, thetas, psis


def concatenate_bin(
    i_bin: int,
    nlsa_movies: list[NDArray[Shape["Any, Any, Any"], Float]],
    taubins_by_prd: list[list[NDArray[Shape["Any"], Int]]],
    phis_by_prd: list[list[NDArray[Shape["Any"], Float]]],
    thetas_by_prd: list[list[NDArray[Shape["Any"], Float]]],
    psis_by_prd: list[list[NDArray[Shape["Any"], Float]]],
    states_per_coord: int,
    pixel_size: int,
    out_dir: str,
):
    imgs, phis, thetas, psis = [], [], [], []

    for i in range(len(nlsa_movies)):
        taubins = taubins_by_prd[i][i_bin]
        if len(taubins) == 0:
            continue
        imgs.append(nlsa_movies[i][taubins][:])
        phis.append(phis_by_prd[i][i_bin])
        thetas.append(thetas_by_prd[i][i_bin])
        psis.append(psis_by_prd[i][i_bin])

    traj_file_rel = f"imgsRELION_{i_bin + 1}_of_{states_per_coord}.mrcs"
    traj_file = os.path.join(out_dir, traj_file_rel)
    ang_file = os.path.join(
        out_dir,
        f"EulerAngles_{i_bin + 1}_of_{states_per_coord}.star",
    )

    imgs = np.concatenate(imgs)
    phis = np.concatenate(phis)
    thetas = np.concatenate(thetas)
    psis = np.concatenate(psis)

    if os.path.exists(traj_file):
        os.remove(traj_file)

    with mrcfile.new(traj_file) as f:
        f.set_data(imgs * -1)

    write_star(ang_file, traj_file_rel, phis, thetas, psis, pixel_size)


def write_relion_s2(
    active_tau_by_prd,
    psi_sorted_quats_by_prd,
    prd_indices: list[int] | NDArray[Shape["Any"], Int],
    tau_avg,
    progress_bar: Any = NullEmitter(),
):
    pathw = params.width_1D
    tqdm = get_tqdm()

    n_prds = len(prd_indices)
    taubins, psis, thetas, phis = (
        [[]] * n_prds,
        [[]] * n_prds,
        [[]] * n_prds,
        [[]] * n_prds,
    )
    for i in tqdm(range(n_prds), "Extracting bin/orientation data..."):
        taubins[i], phis[i], thetas[i], psis[i] = extract_traj_data(
            active_tau_by_prd[i],
            tau_avg,
            psi_sorted_quats_by_prd[i],
            pathw,
            params.states_per_coord,
            params.con_order_range,
        )

    with NamedTemporaryFile(suffix=".h5") as f_tmp:
        with h5pickle.File(f_tmp.name, "w") as f:
            print(f"Using {f_tmp.name} for temporary storage of NLSA data")
            nlsa_builder = partial(
                run_nlsa_second_pass, data_handle=data_store.get_analysis_handle()
            )

            desc = "Generating full NLSA movies..."
            if params.ncpu == 1:
                for i in tqdm(
                    range(n_prds),
                    desc=desc,
                ):
                    f[str(i)] = nlsa_builder(prd_indices[i])
                    progress_bar.emit(int(49 * i / n_prds))
            else:
                with multiprocessing.Pool(processes=params.ncpu) as pool:
                    for i, nlsa_movie in tqdm(
                        enumerate(pool.imap(nlsa_builder, prd_indices)),
                        total=len(prd_indices),
                        desc=desc,
                    ):
                        f[str(i)] = nlsa_movie
                        progress_bar.emit(int(49 * i / n_prds))

            nlsa_movies = [f[str(i)] for i in prd_indices]

            concatenator = partial(
                concatenate_bin,
                nlsa_movies=nlsa_movies,
                taubins_by_prd=taubins,
                phis_by_prd=phis,
                thetas_by_prd=thetas,
                psis_by_prd=psis,
                states_per_coord=params.states_per_coord,
                pixel_size=params.ms_pixel_size,
                out_dir=params.bin_dir,
            )
            bins = range(0, params.states_per_coord - pathw + 1)
            desc = "Outputting RELION stacks and star files..."
            if params.ncpu == 1:
                for i in tqdm(bins, desc=desc):
                    concatenator(i)
                    progress_bar.emit(int(50 + 49 * i / len(bins)))
            else:
                with multiprocessing.Pool(processes=params.ncpu) as pool:
                    for i, _ in tqdm(
                        enumerate(pool.imap_unordered(concatenator, bins)),
                        total=len(bins),
                        desc=desc,
                    ):
                        progress_bar.emit(int(50 + 49 * i / len(bins)))

    progress_bar.emit(100)


def build_trajectory(progress_bar: Any = NullEmitter()):
    """This script prepares the image stacks and orientations for 3D reconstruction."""
    # Copyright (c) UWM, Ali Dashti 2016 (matlab version)
    # Copyright (c) Columbia Univ Hstau Liao 2018 (python version)
    # Copyright (c) Columbia University Suvrajit Maji 2020 (python version)

    params.load()
    active_prds, psinums, senses = data_store.get_active_psinums_and_senses()
    taus = data_store.get_active_taus()
    prds = data_store.get_prds()
    image_indices_by_prd = prds.thresholded_image_indices

    data_handle = data_store.get_analysis_handle()

    active_tau_by_prd = []
    tau_avg = np.array([])
    quats_full = prds.quats_full
    psi_sorted_quats_by_prd: list[NDArray] = []
    for i, i_prd in enumerate(active_prds):
        i_psi = psinums[i]
        psi_sorted_indices = cast(
            NDArray[Shape["Any"], Int],
            cast(
                h5pickle.Dataset,
                data_handle[f"prd_{i_prd}/nlsa_data_{i_psi}/psi_sorted_ind"],
            )[:],
        )
        sorted_raw_indices = image_indices_by_prd[i_prd][psi_sorted_indices]
        psi_sorted_quats_by_prd.append(quats_full[:, sorted_raw_indices])

        tau = taus[i][i_psi]
        if senses[i] == -1:
            tau = 1.0 - tau

        active_tau_by_prd.append(tau)

        tau = tau.flatten()
        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        tau_avg = np.concatenate((tau_avg, tau))

    # Section III
    write_relion_s2(
        active_tau_by_prd,
        psi_sorted_quats_by_prd,
        active_prds,
        tau_avg,
        progress_bar,
    )

    params.project_level = ProjectLevel.TRAJECTORY
    params.save()

    progress_bar.emit(100)
    print(f"Trajectory generation complete! Check {params.bin_dir} for output.")
