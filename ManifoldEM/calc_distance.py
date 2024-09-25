import logging
import mrcfile
import multiprocessing

from typing import List, Union
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.fftpack import ifftshift, fft2, ifft2
from scipy.ndimage import shift, rotate

from nptyping import NDArray, Shape, Float64, Int
from typing import Tuple, Union, List

from ManifoldEM import myio
from ManifoldEM.core import annular_mask, project_mask
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.quaternion import q2Spider, quaternion_to_S2
from ManifoldEM.util import (
    NullEmitter,
    get_tqdm,
    create_proportional_grid,
    get_CTFs,
    rotate_fill,
)

"""
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
"""

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


@dataclass
class FilterParams:
    """
    Class to assist in the creation of image filters.

    Attributes
    ----------
    method : str
        The type of filter, either 'Butter' for Butterworth or 'Gauss' for Gaussian.
    cutoff_freq : float
        The Nyquist cutoff frequency, determining the filter's cutoff threshold.
    order : int
        The filter order, applicable only for the 'Butter' method.

    Methods
    -------
    create_filter(Q)
        Generates a filter based on the specified parameters and a frequency array Q.
    """

    method: str
    cutoff_freq: float
    order: int

    def create_filter(
        self, Q: NDArray[Shape["*,*"], Float64]
    ) -> NDArray[Shape["*,*"], Float64]:
        """
        Creates a filter based on the instance's method, cutoff frequency, and order.

        Parameters
        ----------
        Q : ndarray
            A 2D array of spatial frequencies for which the filter is calculated.

        Returns
        -------
        ndarray
            A 2D array representing the filter in the frequency domain.

        Raises
        ------
        ValueError
            If an unsupported filter method is specified.
        """
        if self.method.lower() == "gauss":
            G = np.exp(-(np.log(2) / 2.0) * (Q / self.cutoff_freq) ** 2)
        elif self.method.lower() == "butter":
            G = np.sqrt(1.0 / (1 + (Q / self.cutoff_freq) ** (2 * self.order)))
        else:
            _logger.error("%s filter is unsupported" % (self.method))
            _logger.exception("%s filter is unsupported" % (self.method))
            raise ValueError("Unsupported filter in distance calculation")

        return G


def get_psi(
    q: NDArray[Shape["4"], Float64], ref_vec: NDArray[Shape["3"], Float64]
) -> float:
    """
    Calculates the psi angle in degrees from a quaternion and a reference vector.

    Parameters
    ----------
    q : ndarray
        A quaternion represented as a 1D NumPy array of shape [4].
    ref_vec : ndarray
        A reference vector represented as a 1D NumPy array of shape [3].

    Returns
    -------
    float
        The psi angle in radians, in the interval [-pi, pi].
    """
    # FIXME: Doc needs way more clarity
    s = -(1 + ref_vec[2]) * q[3] - ref_vec[0] * q[1] - ref_vec[1] * q[2]
    c = (1 + ref_vec[2]) * q[0] + ref_vec[1] * q[1] - ref_vec[0] * q[2]
    if c == 0.0:
        psi = np.sign(s) * np.pi
    else:
        psi = 2 * np.arctan(s / c)  # note that the Psi are in the interval [-pi,pi]

    psi *= 180 / np.pi

    # this happens only for a rotation of pi about an axis perpendicular to the projection direction
    if np.isnan(psi):
        psi = 0.0

    return psi


def psi_ang(ref_vec: NDArray[Shape["3"], Float64]) -> float:
    """
    Calculates the psi angle in degrees from a reference vector.

    Parameters
    ----------
    ref_vec : ndarray
        A reference vector represented as a 1D NumPy array of shape [3].

    Returns
    -------
    float
        The psi angle in degrees.
    """
    # FIXME: Doc needs way more clarity
    Qr = np.array([1 + ref_vec[2], ref_vec[1], -ref_vec[0], 0])
    L2 = np.sum(Qr**2)
    if L2 == 0.0:
        return 0.0
    Qr = Qr / L2
    _, _, psi = q2Spider(Qr)

    psi = np.mod(psi, 2 * np.pi) * (180 / np.pi)
    return psi


def get_wiener(ctf, snr=5.0):
    """
    Computes the Wiener filter domain from a given Contrast Transfer Function (CTF).

    Parameters
    ----------
    ctf : ndarray
        The Contrast Transfer Function represented as a 2D or 3D NumPy array.
    snr : float, default=5.0
        Signal to noise ratio.

    Returns
    -------
    ndarray
        The Wiener filter domain as a NumPy array of the same shape as the input CTF.
    """
    wiener_dom = 0.0
    for i in range(ctf.shape[0]):
        wiener_dom = wiener_dom + ctf[i, :, :] ** 2

    wiener_dom = wiener_dom + 1.0 / snr

    return wiener_dom


@dataclass
class LocalInput:
    """
    A helper class to encapsulate local input data for use with python multiprocessing.

    Attributes
    ----------
    indices : ndarray[int]
        A 1D NumPy array of global image indices.
    quats : ndarray
        A 2D NumPy array (4xN) of rotation quaternions for all images.
    defocus : ndarray
        A 1D NumPy array of defocus values for all images.
    dist_file : str
        Path to the output file where results will be stored.

    Notes
    -----
    This class is designed to organize input data for image processing tasks,
    making it easier to pass multiple related data items as a single object.
    """

    indices: NDArray[Shape["*"], Int]
    quats: NDArray[Shape["4,*"], Float64]
    defocus: NDArray[Shape["*"], Float64]
    dist_file: str


def get_distance_CTF_local(
    input_data: LocalInput,
    filter_params: FilterParams,
    img_file_name: str,
    image_offsets: Tuple[NDArray[Shape["*"], Float64], NDArray[Shape["*"], Float64]],
    relion_data: bool,
):
    """
    This function calculates squared Euclidean distances between images in similar projection directions,
    incorporating Contrast Transfer Function (CTF) correction. It handles both original images and their conjugates,
    effectively doubling the number of data points for analysis.

    Parameters
    ----------
    input_data : LocalInput
         An object containing indices, quaternions, defocus values, and the output file path.
    filter_params : FilterParams
        Parameters for the filter [default gaussian], including cutoff frequency and order.
    img_file_name : str
        Path to the file containing all raw images.
    image_offsets : tuple
        Offsets for each image, typically extracted from STAR files.
    relion_data : bool
        Flag indicating whether the data format is RELION (True) or another format (False).

    The function processes each image based on its index, applying normalization, filtering, and CTF correction.
    It aligns images in-plane using calculated psi angles and computes distances between all pairs of images in the
    given subset. The results, including distances and other relevant data, are saved to the specified output file.

    Notes
    -----
    - Function uses global prd store information, specifically the `image_is_mirrored` array, which is a trick to take prds
    opposite the S2 division plane and place their mirrored version in the appropriate bin, effectively doubling the available
    particles per bin.
    """
    indices = input_data.indices
    quats = input_data.quats
    defocus = input_data.defocus
    out_file = input_data.dist_file

    n_particles = indices.shape[
        0
    ]  # size of bin; ind are the indexes of particles in that bin
    image_is_mirrored = data_store.get_prds().image_is_mirrored

    # auxiliary variables
    n_pix = params.ms_num_pixels

    # different types of averages of aligned particles of the same view
    img_avg = np.zeros((n_pix, n_pix))  # simple average
    img_all = np.zeros((n_particles, n_pix, n_pix))

    fourier_images = np.zeros(
        (n_particles, n_pix, n_pix), dtype=np.complex128
    )  # each (i,:,:) is a Fourier image
    CTF = np.zeros((n_particles, n_pix, n_pix))  # each (i,:,:) is the CTF
    distances = np.zeros(
        (n_particles, n_particles)
    )  # distances among the particles in the bin

    # create grid for filter G
    Q = create_proportional_grid(n_pix)
    G = filter_params.create_filter(Q)
    G = ifftshift(G)

    # reference PR is the average
    avg_orientation_vec = np.sum(quaternion_to_S2(quats), axis=1)
    avg_orientation_vec /= np.linalg.norm(avg_orientation_vec)

    # angle for in-plane rotation alignment
    psi_p = psi_ang(avg_orientation_vec)

    # use volumetric mask, April 2020
    if params.mask_vol_file:
        with mrcfile.open(params.mask_vol_file) as mrc:
            mask3D = mrc.data
        mask = project_mask(mask3D, avg_orientation_vec)
    else:
        mask = annular_mask(0, n_pix / 2.0, n_pix, n_pix)

    if relion_data:
        img_data = mrcfile.mmap(img_file_name, "r").data

    # Total in-plane rotation for each particle
    rotations = np.zeros(n_particles)

    # read images with conjugates
    for i_part in range(n_particles):
        particle_index = indices[i_part]
        if not relion_data:  # spider data
            start = n_pix**2 * particle_index * 4
            img = np.memmap(
                img_file_name,
                dtype="float32",
                offset=start,
                mode="r",
                shape=(n_pix, n_pix),
            ).T
        else:
            shi = (
                image_offsets[1][particle_index] - 0.5,
                image_offsets[0][particle_index] - 0.5,
            )
            img = shift(img_data[particle_index], shi, order=3, mode="wrap")

        # flip images when opposite the S2 division plane
        if image_is_mirrored[particle_index]:
            img = np.flipud(img)

        # Apply the filter
        img = ifft2(fft2(img) * G).real

        # Get the psi angle
        rotations[i_part] = -get_psi(quats[:, i_part], avg_orientation_vec) - psi_p

        # inplane align the images
        img = rotate_fill(img, rotations[i_part])

        # Apply mask and store for distance calculation
        img_all[i_part, :, :] = img * mask

    CTF = get_CTFs(
        params.ms_num_pixels,
        defocus,
        params.ms_spherical_aberration,
        params.ms_kilovolts,
        params.ms_ctf_envelope,
        params.ms_amplitude_contrast_ratio,
    )

    # use wiener filter
    img_avg = np.zeros((n_pix, n_pix))
    wiener_dom = -get_wiener(CTF)
    for i_part in range(n_particles):
        img = img_all[i_part, :, :]
        img = (img - img.mean()) / img.std()
        img_f = fft2(img)
        fourier_images[i_part, :, :] = img_f
        CTF_i = CTF[i_part, :, :]
        img_f_wiener = img_f * (CTF_i / wiener_dom)
        img_avg = img_avg + ifft2(img_f_wiener).real

    # plain and phase-flipped averages
    # April 2020, msk2 = 1 when there is no volume mask
    img_avg = img_avg / n_particles

    fourier_images = fourier_images.reshape(n_particles, n_pix**2)
    CTF = CTF.reshape(n_particles, n_pix**2)

    # fancy BLAS way to do D[i,j] = np.sum(np.abs(CTF[i, :] * fourier_image[j,:] - CTF[j, :] * fourier_image[i, :])**2)
    # FIXME (RB): Numba could, in theory, avoid all of these large temporaries, but I was struggling to make it actually work efficiently
    # The test C++ code is considerably faster with no temporaries and large numbers of particles, so should revisit
    CTFfy = CTF.conj() * fourier_images
    distances = np.dot((np.abs(CTF) ** 2), (np.abs(fourier_images) ** 2).T)
    distances = (
        distances + distances.T - 2 * np.real(np.dot(CTFfy, CTFfy.conj().transpose()))
    )

    distances[np.diag_indices(n_particles)] = 0.0

    myio.fout1(
        out_file,
        D=distances,
        ind=indices,
        q=quats,
        msk2=mask.astype(np.float16),
        PD=avg_orientation_vec,
        imgAvg=img_avg.astype(np.float16),
        rotations=rotations,
        image_filter=G,
    )


def _construct_input_data(prd_list, thresholded_indices, quats_full, defocus):
    """
    Constructs a list of LocalInput objects from given indices, quaternions, and defocus values.

    Parameters
    ----------
    prd_list : Union[None, List[int]]
        List of prds to process. If `None`, use `thresholded_indices`, otherwise use the intersection of `prd_list` and
        `thresholded_indices`.
    thresholded_indices : List[int]
        Indices of prds that meet the user input threshold requirements from earlier in the Manifold pipeline.
    quats_full : ndarray
        A 2D array of shape [4, N] containing the rotation quaternions for all prds.
    defocus : ndarray
        A 1D array containing the defocus values for all prds.

    Returns
    -------
    List[LocalInput]
        A list of LocalInput objects for processing (one for each prd).
    """
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
        ll.append(
            LocalInput(ind, quats_full[:, ind], defocus[ind], params.get_dist_file(prD))
        )

    return ll


def op(prd_list: Union[List[int], None] = None, *argv):
    """
    This function calculates squared Euclidean distances between images within each prd bin,
    incorporating Contrast Transfer Function (CTF) correction. `params.ncpu` prds are run
    in parallel.

    Parameters
    ----------
    prd_list : Union[List[int], None], default=None
        List of prds to process. If `None`, use `thresholded_indices`, otherwise use the intersection of `prd_list` and
        `thresholded_indices`. When `None`, also increments the `params.project_level` when complete.
    *argv : tuple
        If another argument is supplied, it's assumed to be for progress tracking when using the QT gui. Not for users.
    """
    print("Computing the distances...")
    params.load()
    multiprocessing.set_start_method("fork", force=True)
    use_gui_progress = len(argv) > 0

    prds = data_store.get_prds()

    filter_params = FilterParams(
        method=params.distance_filter_type,
        cutoff_freq=params.distance_filter_cutoff_freq,
        order=params.distance_filter_order,
    )

    input_data = _construct_input_data(
        prd_list, prds.thresholded_image_indices, prds.quats_full, prds.defocus
    )
    n_jobs = len(input_data)
    local_distance_func = partial(
        get_distance_CTF_local,
        filter_params=filter_params,
        img_file_name=params.img_stack_file,
        image_offsets=prds.microscope_origin,
        relion_data=params.is_relion_data,
    )

    progress1 = argv[0] if use_gui_progress else NullEmitter()
    tqdm = get_tqdm()
    if params.ncpu == 1:
        for i, datai in tqdm(
            enumerate(input_data), total=n_jobs, disable=use_gui_progress
        ):
            local_distance_func(datai)
            progress1.emit(int(99 * i / n_jobs))
    else:
        with multiprocessing.Pool(processes=params.ncpu) as pool:
            for i, _ in tqdm(
                enumerate(pool.imap_unordered(local_distance_func, input_data)),
                total=n_jobs,
                disable=use_gui_progress,
            ):
                progress1.emit(int(99 * i / n_jobs))

    if prd_list is None:
        params.project_level = ProjectLevel.CALC_DISTANCE

    params.save()
    progress1.emit(100)
