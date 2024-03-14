import logging
import mrcfile
import multiprocessing
import tqdm

from dataclasses import dataclass
from functools import partial

import numpy as np
from PIL import Image
from scipy.fftpack import ifftshift, fft2, ifft2
from scipy.ndimage import shift

from nptyping import NDArray, Shape, Float64, Int
from typing import Tuple

from ManifoldEM import myio, projectMask
from ManifoldEM.core import annularMask
from ManifoldEM.data_store import data_store
from ManifoldEM.params import p
from ManifoldEM.quaternion import q2Spider
from ManifoldEM.util import NullEmitter
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


@dataclass
class FilterParams:
    """
    Class to assist in creation of image filters
    method:      'Butter' or 'Gauss'
    cutoff_freq: Nyquist cutoff freq
    order:       Filter order (for 'Butter' only)
    """
    method: str
    cutoff_freq: float
    order: int

    def create_filter(self, Q: NDArray[Shape["*,*"], Float64]) -> NDArray[Shape["*,*"], Float64]:
        if self.method == 'Gauss':
            G = np.exp(-(np.log(2) / 2.) * (Q / self.cutoff_freq)**2)
        elif self.method == 'Butter':
            G = np.sqrt(1. / (1 + (Q / self.cutoff_freq)**(2 * self.order)))
        else:
            _logger.error('%s filter is unsupported' % (self.method))
            _logger.exception('%s filter is unsupported' % (self.method))
            raise ValueError

        return G


def rotate_fill(img: NDArray[Shape["*,*"], Float64], angle: float) -> NDArray[Shape["*,*"], Float64]:
    n_pix = img.shape[0]
    in_rep = Image.fromarray(np.tile(img, (3, 3)).astype('float32'), mode='F')
    out_rep = np.array(in_rep.rotate(angle, expand=False), dtype=img.dtype)
    return out_rep[n_pix:2 * n_pix, n_pix:2 * n_pix]


def create_grid(N: int) -> NDArray[Shape["*,*"], Float64]:
    """Create NxN grid centered grid around (0, 0)"""
    a = np.arange(N) - N // 2
    X, Y = np.meshgrid(a, a)

    return 2 * np.sqrt(X**2 + Y**2) / N


def quats_to_unit_vecs(q: NDArray[Shape["4,*"], Float64]) -> NDArray[Shape["3,*"], Float64]:
    # Calculate average projection directions (from matlab code)
    PDs = 2 * np.vstack((q[1, :] * q[3, :] - q[0, :] * q[2, :], q[0, :] * q[1, :] + q[2, :] * q[3, :],
                         q[0, :]**2 + q[3, :]**2 - np.ones((1, q.shape[1])) / 2.0))

    return PDs


def get_psi(q: NDArray[Shape["4"], Float64], ref_vec: NDArray[Shape["3"], Float64]) -> float:
    s = -(1 + ref_vec[2]) * q[3] - ref_vec[0] * q[1] - ref_vec[1] * q[2]
    c = (1 + ref_vec[2]) * q[0] + ref_vec[1] * q[1] - ref_vec[0] * q[2]
    if c == 0.0:
        psi = np.sign(s) * np.pi
    else:
        psi = 2 * np.arctan(s / c)  # note that the Psi are in the interval [-pi,pi]

    return psi


def psi_ang(ref_vec: NDArray[Shape["3"], Float64]):
    Qr = np.array([1 + ref_vec[2], ref_vec[1], -ref_vec[0], 0])
    L2 = np.sum(Qr**2)
    if L2 == 0.0:
        return 0.0
    Qr = Qr / L2
    _, _, psi = q2Spider(Qr)

    psi = np.mod(psi, 2 * np.pi) * (180 / np.pi)
    return psi


def get_wiener1(CTF1):
    SNR = 5
    wiener_dom = 0.
    for i in range(CTF1.shape[0]):
        wiener_dom = wiener_dom + CTF1[i, :, :]**2

    wiener_dom = wiener_dom + 1. / SNR

    return (wiener_dom)


def ctemh_cryoFrank(k: NDArray[Shape["*,*"], Float64], spherical_aberration: float, defocus: float,
                    electron_energy: float, gauss_env_halfwidth: float, amplitude_contrast_ratio: float):
    """
    from Kirkland, adapted for cryo (EMAN1) by P. Schwander
    Version V 1.1
    Copyright (c) UWM, Peter Schwander 2010 MATLAB version

    Copyright (c) Columbia University Hstau Liao 2018 (python version)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Here, the damping envelope is characterized by a single parameter B (gauss_env)
    see J. Frank
    spherical_abberation (Cs) in mm
    defocus (df) in Angstrom, a positive value is underfocus
    electron_energy in keV
    gauss_env (B) in A^-2
    amplitude_constrast_ratio from alignment file

    Note: we assume |k| = s
    """
    spherical_aberration *= 1.0e7
    mo = 511.0
    hc = 12.3986
    wav = (2 * mo) + electron_energy
    wav = hc / np.sqrt(wav * electron_energy)
    w1 = np.pi * spherical_aberration * wav * wav * wav
    w2 = np.pi * wav * defocus
    k2 = k * k
    sigm = gauss_env_halfwidth / np.sqrt(2 * np.log(2))
    wi = np.exp(-k2 / (2 * sigm**2))
    wr = (0.5 * w1 * k2 - w2) * k2  # gam = (pi/2)Cs lam^3 k^4 - pi lam df k^2

    return (np.sin(wr) - amplitude_contrast_ratio * np.cos(wr)) * wi


@dataclass
class LocalInput:
    indices: NDArray[Shape["*"], Int]     #  Global image indexes
    quats: NDArray[Shape["4,*"], Float64] #  Rotation quaternions of all images, 4xN
    defocus: NDArray[Shape["*"], Float64] #  Defocus values of all images
    dist_file: str                        #  Output file for results


def get_distance_CTF_local(input_data: LocalInput, filter_params: FilterParams, img_file_name: str,
                           image_offsets: Tuple[NDArray[Shape["*"], Float64], NDArray[Shape["*"], Float64]],
                           n_particles_tot: int, relion_data: bool):
    """
    Calculates squared Euclidian distances for snapshots in similar
    projection directions. Includes CTF correction of microscope.
    Version with conjugates, effectively double number of data points

    Input parameters
    input_data     see LocalInput
    filter_params  Filter Gaussian width [pixel]
    image_offsets  Image origins (from star files, usually. aka "sh")
    img_file_name  Image file with all raw images

    Uses the following microscope data from global params:
        Cs         Spherical aberration [mm]
        EkV        Acceleration voltage [kV]
        gaussEnv   Gaussian damping envelope [A^-1]
        nPix       lateral pixel count
        dPix       Pixel size [A]
    """
    indices = input_data.indices
    quats = input_data.quats
    defocus = input_data.defocus
    out_file = input_data.dist_file

    n_particles = indices.shape[0]  # size of bin; ind are the indexes of particles in that bin
    # auxiliary variables
    n_pix = p.nPix

    # different types of averages of aligned particles of the same view
    img_avg = np.zeros((n_pix, n_pix))  # simple average
    img_all = np.zeros((n_particles, n_pix, n_pix))

    fourier_images = np.zeros((n_particles, n_pix, n_pix), dtype=np.complex128)  # each (i,:,:) is a Fourier image
    CTF = np.zeros((n_particles, n_pix, n_pix))  # each (i,:,:) is the CTF
    distances = np.zeros((n_particles, n_particles))  # distances among the particles in the bin

    msk = annularMask(0, n_pix / 2., n_pix, n_pix)

    # create grid for filter G
    Q = create_grid(n_pix)
    G = filter_params.create_filter(Q)
    G = ifftshift(G)

    # reference PR is the average
    avg_orientation_vec = np.sum(quats_to_unit_vecs(quats), axis=1)
    avg_orientation_vec /= np.linalg.norm(avg_orientation_vec)

    # psi_p angle for in-plane rotation alignment
    psi_p = psi_ang(avg_orientation_vec)

    # use volumetric mask, April 2020
    if p.mask_vol_file:
        with mrcfile.open(p.mask_vol_file) as mrc:
            mask3D = mrc.data
        msk2 = projectMask.op(mask3D, avg_orientation_vec)
    else:
        msk2 = 1

    # read images with conjugates
    for i_part in range(n_particles):
        if indices[i_part] < n_particles_tot / 2:  # first half data set; i.e., before augmentation
            raw_particle_index = indices[i_part]
        else:  # second half data set; i.e., the conjugates
            raw_particle_index = int(indices[i_part] - n_particles_tot / 2)
        if not relion_data:  # spider data
            start = n_pix**2 * raw_particle_index * 4
            img = np.memmap(img_file_name, dtype='float32', offset=start, mode='r', shape=(n_pix, n_pix)).T
        else:  # relion data
            img = mrcfile.mmap(img_file_name, 'r').data[raw_particle_index]
            shi = (image_offsets[1][raw_particle_index] - 0.5, image_offsets[0][raw_particle_index] - 0.5)
            img = shift(img, shi, order=3, mode='wrap')
        if indices[i_part] >= n_particles_tot / 2:  # second half data set
            img = np.flipud(img)

        # normalizing
        backg = img * (1 - msk)
        std = backg.std()
        if not std == 0.:
            img = (img - backg.mean()) / std
        else:
            print(f"Warning: flat image found at index: {raw_particle_index}")

        # store each flatted image in y and filter
        img = img.flatten('F')
        img = img.reshape(-1, n_pix).transpose()
        img = ifft2(fft2(img) * G).real.flatten('F')

        # Get the psi angle
        psi = get_psi(quats[:, i_part], avg_orientation_vec)

        # this happens only for a rotation of pi about an axis perpendicular to the projection direction
        if np.isnan(psi):
            psi = 0.

        # inplane align the images
        img = img.reshape(-1, n_pix).transpose() * msk  # convert to matlab convention prior to rotation
        img = rotate_fill(img, -(180 / np.pi) * psi)
        img = rotate_fill(img, -psi_p)

        # CTF info
        ctf_i = ctemh_cryoFrank(Q / (2 * p.pix_size), p.Cs, defocus[i_part], p.EkV, p.gaussEnv, p.AmpContrast)
        CTF[i_part, :, :] = ifftshift(ctf_i)

        # Fourier transformed #April 2020, with vol mask msk2, used for distance calc D
        fourier_images[i_part, :, :] = fft2(img * msk2)

        img_all[i_part, :, :] = img

    # use wiener filter
    img_avg = 0
    wiener_dom = -get_wiener1(CTF)
    for i_part in range(n_particles):
        img = img_all[i_part, :, :]
        img_f = fft2(img)  #.reshape(dim, dim)) T only for matlab
        CTF_i = CTF[i_part, :, :]
        img_f_wiener = img_f * (CTF_i / wiener_dom)
        img_avg = img_avg + ifft2(img_f_wiener).real

    # plain and phase-flipped averages
    # April 2020, msk2 = 1 when there is no volume mask
    img_avg = img_avg * msk2 / n_particles

    fourier_images = fourier_images.reshape(n_particles, n_pix**2)
    CTF = CTF.reshape(n_particles, n_pix**2)

    CTFfy = CTF.conj() * fourier_images
    distances = np.dot((np.abs(CTF)**2), (np.abs(fourier_images)**2).T)
    distances = distances + distances.T - 2 * np.real(np.dot(CTFfy, CTFfy.conj().transpose()))

    myio.fout1(out_file,
               D=distances,
               ind=indices,
               q=quats,
               CTF=CTF,
               imgAll=img_all,
               msk2=msk2,
               PD=avg_orientation_vec,
               imgAvg=img_avg)


def _construct_input_data(prd_list, thresholded_indices, quats_full, defocus):
    n_prds = len(thresholded_indices)
    valid_prds = set(range(n_prds))
    if prd_list is not None:
        requested_prds = set(prd_list)
        invalid_prds = requested_prds.difference(valid_prds)
        if invalid_prds:
            print(f"Warning: requested invalid prds: {invalid_prds}")
        valid_prds = valid_prds.intersection(requested_prds)

    ll = []
    for prD in valid_prds:
        ind = thresholded_indices[prD]
        ll.append(LocalInput(ind, quats_full[:, ind], defocus[ind], p.get_dist_file(prD)))

    return ll


def op(prd_list: Union[List[int], None], *argv):
    print("Computing the distances...")
    p.load()
    multiprocessing.set_start_method('fork', force=True)
    use_gui_progress = len(argv) > 0

    prds = data_store.get_prds()

    filter_params = FilterParams(method='Butter', cutoff_freq=0.5, order=8)

    input_data = _construct_input_data(prd_list, prds.thresholded_image_indices, prds.quats_full, prds.defocus)
    n_jobs = len(input_data)
    local_distance_func = partial(get_distance_CTF_local,
                                  filter_params=filter_params,
                                  img_file_name=p.img_stack_file,
                                  image_offsets=prds.microscope_origin,
                                  n_particles_tot=len(prds.defocus),
                                  relion_data=p.relion_data)

    progress1 = argv[0] if use_gui_progress else NullEmitter()

    if p.ncpu == 1:
        for i, datai in tqdm.tqdm(enumerate(input_data), total=n_jobs, disable=use_gui_progress):
            local_distance_func(datai)
            progress1.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(local_distance_func, input_data)),
                                  total=n_jobs,
                                  disable=use_gui_progress):
                progress1.emit(int(99 * i / n_jobs))

    p.save()
    progress1.emit(100)
