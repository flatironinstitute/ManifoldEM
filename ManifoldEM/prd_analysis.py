import h5py
import logging
import mrcfile
import multiprocessing
import numpy as np

from dataclasses import dataclass
from typing import Any, Tuple, Iterable
from nptyping import NDArray, Shape, Float, Int, Bool
from numpy.fft import fft2, ifft2, ifftshift
from scipy.ndimage import shift

from ManifoldEM.core import annular_mask, project_mask, svdRF, L2_distance, get_wiener
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params
from ManifoldEM.quaternion import quaternion_to_S2, q2Spider
from ManifoldEM.util import create_proportional_grid, get_CTFs, rotate_fill, get_tqdm
from ManifoldEM.DMembeddingII import diffusion_map_embedding
from ManifoldEM.fit_1D_open_manifold_3D import fit_1D_open_manifold_3D
from ManifoldEM.util import recursive_dict_to_hdf5

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
        self, Q: NDArray[Shape["Any,Any"], Float]
    ) -> NDArray[Shape["Any,Any"], Float]:
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


def get_wiener_basic(ctf, snr=5.0):
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


def psi_ang(ref_vec: NDArray[Shape["3"], Float]) -> float:
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


def get_psi(
    q: NDArray[Shape["4"], Float], ref_vec: NDArray[Shape["3"], Float]
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


def calc_distances(
    images: NDArray[Shape["Any,Any,Any"], Float],
    CTF: NDArray[Shape["Any,Any,Any"], Float],
):
    n_particles = images.shape[0]
    n_pix = images.shape[1]

    # use wiener filter
    wiener_img_avg = np.zeros((n_pix, n_pix))
    wiener_dom = -get_wiener_basic(CTF)
    fourier_images = np.zeros((n_particles, n_pix, n_pix), dtype=np.complex128)
    for i_part in range(n_particles):
        img = images[i_part, :, :]
        img = (img - img.mean()) / img.std()
        img_f = fft2(img)
        fourier_images[i_part, :, :] = img_f
        CTF_i = CTF[i_part, :, :]
        img_f_wiener = img_f * (CTF_i / wiener_dom)
        wiener_img_avg = wiener_img_avg + ifft2(img_f_wiener).real

    # plain and phase-flipped averages
    # April 2020, msk2 = 1 when there is no volume mask
    wiener_img_avg = wiener_img_avg / n_particles

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

    return dict(D=distances, imgAvg=wiener_img_avg)


def get_transform_info(
    n_pix: int,
    quats: NDArray[Shape["Any,4"], Float],
    filter: FilterParams,
    mask_vol_file: str = "",
) -> Tuple[
    NDArray[Shape["Any,Any"], Float],
    NDArray[Shape["Any,Any"], Float],
    NDArray[Shape["Any"], Float],
    NDArray[Shape["3"], Float],
]:
    if len(quats.shape) != 2 or quats.shape[1] != 4:
        raise ValueError("Quaternions must be of shape (N, 4)")

    n_images = quats.shape[0]

    Q = create_proportional_grid(n_pix)
    G = ifftshift(filter.create_filter(Q))

    avg_orientation_vec = np.sum(quaternion_to_S2(quats.T), axis=1)
    avg_orientation_vec /= np.linalg.norm(avg_orientation_vec)
    psi_p = psi_ang(avg_orientation_vec)

    if mask_vol_file:
        with mrcfile.open(mask_vol_file) as mrc:
            mask3D = mrc.data
            mask = project_mask(mask3D, avg_orientation_vec)
    else:
        mask = annular_mask(0, n_pix / 2.0, n_pix, n_pix)

    rotations = np.empty(n_images)
    for i_image in range(n_images):
        rotations[i_image] = -get_psi(quats[i_image, :], avg_orientation_vec) - psi_p

    return G, mask, rotations, avg_orientation_vec


def get_raw_images(
    image_stack_file: str, image_indices: NDArray[Shape["Any"], Int]
) -> NDArray[Shape["Any,Any,Any"], Float]:
    """
    Retrieves a subset of images from an mrcs image stack file.

    Parameters
    ----------
    image_stack_file : str
        The path to the image stack file
    image_indices : ndarray
        The indices of the images to retrieve

    Returns
    -------
    ndarray
        A 3D NumPy array of shape (N, H, W) containing the images,
        where N is the number of images and H and W are the image dimensions.

    Raises
    ------
    ValueError
        If the image stack is not a NumPy array or is not 3D.
    """
    with mrcfile.mmap(image_stack_file, "r") as mrc:
        images = mrc.data

        if not isinstance(images, (np.memmap, np.ndarray)):
            raise ValueError("Image stack must be a NumPy array")
        if len(images.shape) != 3:
            raise ValueError("Image stack must be 3D")

    return images[image_indices, :, :]


def transform_images(
    raw_images: NDArray[Shape["Any,Any,Any"], Float],
    image_filter: NDArray[Shape["Any,Any"], Float],
    mask: NDArray[Shape["Any,Any"], Float],
    offsets: NDArray[Shape["Any,2"], Float],
    rotations: NDArray[Shape["Any"], Float],
    mirror: NDArray[Shape["Any"], Bool],
    in_place: bool = False,
):
    n_images = raw_images.shape[0]

    transformed_images = raw_images if in_place else np.copy(raw_images)
    for i in range(n_images):
        transformed_images[i] = shift(
            transformed_images[i],
            (offsets[i, 0] - 0.5, offsets[i, 1] - 0.5),
            order=3,
            mode="wrap",
        )

        if mirror[i]:
            transformed_images[i] = np.flipud(transformed_images[i])

        transformed_images[i] = ifft2(fft2(transformed_images[i]) * image_filter).real
        transformed_images[i] = rotate_fill(transformed_images[i], rotations[i])
        transformed_images[i] = transformed_images[i] * mask

    return transformed_images


def get_psiPath(psi, rad, plotNum):
    """
    Identifies the indices of points in the embedded space within a specified radius from the origin.

    Parameters
    ----------
    psi : ndarray
        The matrix of embedded coordinates, where each column represents a dimension.
    rad : float
        The radius within which to consider points as being part of the path.
    plotNum : int
        The starting dimension in the embedded space for calculating distances.

    Returns
    -------
    ndarray
        An array of indices of points within the specified radius from the origin in the
        embedded space defined by the dimensions starting at plotNum.

    Notes:
    - This function calculates the Euclidean distance from the origin in a 3-dimensional space
      defined by the dimensions plotNum, plotNum+1, and plotNum+2 of the embedded coordinates.
    - Points with a distance less than 'rad' from the origin are considered part of the path.
    """
    psinum1 = plotNum
    psinum2 = plotNum + 1
    psinum3 = plotNum + 2

    psiDist = np.sqrt(
        psi[:, psinum1] ** 2 + psi[:, psinum2] ** 2 + psi[:, psinum3] ** 2
    )

    posPathInt = (psiDist < rad).nonzero()[0]

    return posPathInt


def auto_trim_manifold(distances, nlsa_tune: int, rad: float):
    n_images = nS = distances.shape[0]
    distances = np.copy(distances)
    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = diffusion_map_embedding(
        distances, distances.shape[0], nlsa_tune
    )
    posPath1 = get_psiPath(psi, rad, 0)

    while len(posPath1) < nS:
        nS = len(posPath1)
        D1 = distances[posPath1][:, posPath1]
        lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = (
            diffusion_map_embedding(D1, D1.shape[0], nlsa_tune)
        )
        lamb = lamb[lamb > 0]
        posPathInt = get_psiPath(psi, rad, 0)
        posPath1 = posPath1[posPathInt]

    posPath = np.arange(n_images)[posPath1]
    return dict(
        lamb=lamb,
        psi=psi,
        sigma=sigma,
        mu=mu,
        posPath=posPath,
        logEps=logEps,
        logSumWij=logSumWij,
        popt=popt,
        R_squared=R_squared,
    )


def NLSA(
    distance_mat: NDArray[Shape["Any,Any"], Float],
    image_indices: NDArray[Shape["Any"], Int],
    posPsi1: NDArray[Shape["Any"], Int],
    all_images: NDArray[Shape["Any,Any,Any"], Float],
    mask: NDArray[Shape["Any,Any"], Float],
    CTF: NDArray[Shape["Any,Any,Any"], Float],
    con_order: int,
    tune: float,
    psiTrunc: int,
):
    n_images = distance_mat.shape[0]
    block_size = n_images - con_order
    con_dist_mat = np.zeros((block_size, block_size))
    # convolve distance matrix with identity along diagonal (like a blur filter)
    for i in range(con_order):
        ind = range(i, i + block_size)
        con_dist_mat += distance_mat[ind][:, ind]

    # find the manifold mapping:
    _, psiC, _, mu, _, _, _, _ = diffusion_map_embedding(con_dist_mat, block_size, tune)

    psiC1 = np.copy(psiC)
    # Wiener filtering
    wiener_dom, CTF1 = get_wiener(CTF, image_indices, posPsi1, con_order, n_images)

    n_pixels = CTF.shape[1] ** 2
    ell = psiTrunc - 1
    psiC = np.hstack((np.ones((psiC.shape[0], 1)), psiC[:, 0:ell]))
    mu_psi = mu.reshape((-1, 1)) * psiC
    A = np.zeros((con_order * n_pixels, ell + 1), dtype=np.float64)
    tmp = np.zeros((n_pixels, block_size), dtype=np.float64)

    img_f_wiener = np.zeros(CTF[0].shape, dtype=np.complex128)
    img = np.zeros(CTF[0].shape, dtype=np.float64)
    image_indices_psi_sorted = image_indices[posPsi1]
    for ii in range(con_order):
        for i in range(block_size):
            ind = con_order - ii + i - 1
            img[:] = all_images[image_indices_psi_sorted[ind]]
            img_f_wiener[:] = fft2(img) * CTF1[ind] / wiener_dom[i]
            img[:] = ifft2(img_f_wiener).real * mask
            tmp[:, i] = np.squeeze(img.T.ravel())

        A[ii * n_pixels : (ii + 1) * n_pixels, :] = tmp @ mu_psi

    U, S, V = svdRF(A)
    VX = V.T @ psiC.T

    sdiag = np.diag(S)

    Topo_mean = np.zeros((n_pixels, psiTrunc))
    for ii in range(psiTrunc):  # of topos considered
        # s = s + 1  needed?
        Topo = np.ones((n_pixels, con_order)) * np.Inf

        for block_size in range(con_order):
            Topo[:, block_size] = U[
                block_size * n_pixels : (block_size + 1) * n_pixels, ii
            ]

        Topo_mean[:, ii] = np.mean(Topo, axis=1)

    # unwrapping... REVIEW; allow user option to select from a list of chronos ([0,1,3]) to retain (i.e., not just [0, 1])
    ConImgT = np.zeros((max(U.shape), ell + 1), dtype=np.float64)
    for i in range(0, 2):
        ConImgT = ConImgT + np.matmul(
            U[:, i].reshape(-1, 1), sdiag[i] * (V[:, i].reshape(1, -1))
        )

    rec_num = con_order
    img_shape = all_images.shape[1:]
    IMGT = np.zeros((n_images - con_order - rec_num, *img_shape), dtype=np.float64)
    for i in range(rec_num):
        tmp = ConImgT[i * n_pixels : (i + 1) * n_pixels, :] @ psiC.T
        for ii in range(n_images - 2 * con_order):
            IMGT[ii] += tmp[:, i + ii].reshape(img_shape)

    # normalize per frame so that mean=0 std=1, whole frame (this needs justif)
    for i in range(IMGT.shape[0]):
        ttmp = IMGT[i, :, :]
        try:
            ttmp = (ttmp - np.mean(ttmp)) / np.std(ttmp)
        except Exception as e:
            msg = f"Error in NLSA normalization. Flat image?\nOriginal exception: {e}"
            raise ValueError(msg)

        IMGT[i, :, :] = ttmp

    nSrecon = IMGT.shape[0]
    Drecon = L2_distance(IMGT, IMGT)

    lamb, psirec, _, mu, _, _, _, _ = diffusion_map_embedding(
        (Drecon**2), nSrecon, tune
    )

    lamb = lamb[lamb > 0]
    _, _, tau = fit_1D_open_manifold_3D(psirec)

    # tau is #part (num-2ConOrder?)
    # psirec is #part x #eigs
    return (IMGT, Topo_mean, psirec, psiC1, sdiag, VX, mu, tau)


def psi_analysis_single(
    distances,
    images,
    CTF,
    mask,
    psi,
    psi_list,
    img_indices,
    con_order_range,
    psi_trunc,
):
    n_images = len(img_indices)  # number of images in PD
    con_order = n_images // con_order_range
    # if ConOrder is large, noise-free 2D frames expected w/ small range of conformations, \
    # while losing snapshots at edges

    img_indices = np.squeeze(np.array(img_indices))
    distances = distances[img_indices][:, img_indices]

    res = []
    for psinum in psi_list:  # for each reaction coordinates do the following:
        if psinum == -1:
            continue

        # e.g., shape=(numPDs,): reordering image indices along each diff map coord
        psi_sorted_ind = np.argsort(psi[:, psinum])
        # distance matrix with indices of images re-arranged along current diffusion map coordinate
        DD = distances[psi_sorted_ind][:, psi_sorted_ind]
        nlsa_images, topo_mean, psirec, psiC1, sdiag, VX, mu, tau = NLSA(
            DD,
            img_indices,
            psi_sorted_ind,
            images,
            mask,
            CTF,
            con_order=con_order,
            tune=params.nlsa_tune,
            psiTrunc=psi_trunc,
        )

        n_s_recon = nlsa_images.shape[0]
        numclass = min(params.states_per_coord, n_s_recon // 2)

        tau = (tau - min(tau)) / (max(tau) - min(tau))

        IMG1 = np.zeros((numclass, *nlsa_images.shape[1:]), dtype=np.float16)
        tauinds = np.empty(numclass, dtype=np.int32)
        for i in range(numclass):
            ind1 = float(i) / numclass
            ind2 = ind1 + 1.0 / numclass
            if i == numclass - 1:
                tauind = ((tau >= ind1) & (tau <= ind2)).nonzero()[0]
            else:
                tauind = ((tau >= ind1) & (tau < ind2)).nonzero()[0]
            while tauind.size == 0:
                sc = 1.0 / (numclass * 2.0)
                ind1 = ind1 - sc * ind1
                ind2 = ind2 + sc * ind2
                tauind = ((tau >= ind1) & (tau < ind2)).nonzero()[0]

            IMG1[i] = nlsa_images[tauind[0]]
            tauinds[i] = tauind[0]

        res.append(
            dict(
                IMG1=IMG1.astype(np.float16),
                psirec=psirec,
                tau=tau,
                psiC1=psiC1,
                mu=mu,
                VX=VX,
                sdiag=sdiag,
                Topo_mean=topo_mean.astype(np.float16),
                tauinds=tauinds,
            )
        )

    return res


def run_pipeline(prd_index: int):
    prds = data_store.get_prds()
    raw_image_indices = prds.thresholded_image_indices[prd_index]
    image_mirrored = prds.image_is_mirrored[raw_image_indices]
    image_defocus = prds.get_defocus_by_prd(prd_index)
    image_offsets = np.empty((len(raw_image_indices), 2))
    image_offsets[:, 0] = prds.microscope_origin[1][raw_image_indices]
    image_offsets[:, 1] = prds.microscope_origin[0][raw_image_indices]

    filter_params = FilterParams(
        method=params.distance_filter_type,
        cutoff_freq=params.distance_filter_cutoff_freq,
        order=params.distance_filter_order,
    )

    image_quats = prds.quats_full[:, raw_image_indices].T
    (image_filter, image_mask, image_rotations, avg_orientation) = get_transform_info(
        params.ms_num_pixels,
        image_quats,
        filter_params,
        mask_vol_file=params.mask_vol_file,
    )

    raw_images = get_raw_images(params.img_stack_file, raw_image_indices)
    transformed_images = transform_images(
        raw_images,
        image_filter,
        image_mask,
        image_offsets,
        image_rotations,
        image_mirrored,
        in_place=True,
    )
    image_CTFs = get_CTFs(
        image_defocus,
        params.ms_num_pixels,
        params.ms_spherical_aberration,
        params.ms_kilovolts,
        params.ms_ctf_envelope,
        params.ms_amplitude_contrast_ratio,
    )

    distances_data = calc_distances(transformed_images, image_CTFs)
    embed_data = auto_trim_manifold(distances_data["D"], params.nlsa_tune, params.rad)
    nlsa_data = psi_analysis_single(
        distances_data["D"],
        transformed_images,
        image_CTFs,
        image_mask,
        embed_data["psi"],
        np.arange(params.num_psi),
        embed_data["posPath"],
        params.con_order_range,
        params.num_psi_truncated,
    )

    image_data = dict(
        filter=image_filter,
        mask=image_mask,
        rotations=image_rotations,
        avg_orientation=avg_orientation,
    )
    res = {
        "image_data": image_data,
        "distances": distances_data,
        "embedding_data": embed_data,
        "nlsa_data": nlsa_data,
    }

    return res


def prd_analysis(
    project_file: None | str,
    prd_indices: None | Iterable[int] = None,
    return_output: bool = True,
    output_handle: None | h5py.Group = None,
    ncpu: int = 1,
):
    """Runs the preprocessing analysis pipeline for a set of projection directions.

    Parameters
    ----------
    project_file : str
        The path to the project file containing the parameters
    prd_indices : None | Iterable[int], optional
        The indices of the projection directions to analyze. If None, all active projection directions are analyzed
    return_output : bool, optional
        Whether to return the output data. More memory intensive, but allows direct manipulation of return data
    output_handle : None | h5py.Group, optional
        The HDF5 group to write the output data to. If None, no output data is written

    Returns
    -------
    dict[int, Any] | None
        A dictionary of the output data for each projection direction, indexed by the projection direction index.
        If return_output is False, this is None.

    Raises
    ------
    ValueError
        If neither return_output nor output_handle is specified.
    """
    if not return_output and not output_handle:
        raise ValueError("Either return_output or output_handle must be specified")

    if project_file is not None:
        params.load(project_file)

    prds = data_store.get_prds()
    if prd_indices is None:
        prd_indices = list(range(prds.n_thresholded))
    else:
        prd_indices = list(prd_indices)

    tqdm = get_tqdm()
    prd_data: dict[int, Any] = {}

    if ncpu == 1:
        for i in tqdm(
            prd_indices,
            total=len(prd_indices),
            desc="Running manifold decomposition...",
        ):
            result = run_pipeline(i)
            if return_output:
                prd_data[i] = result
            if output_handle:
                group_name = f"prd_{i}"
                if output_handle.get(group_name):
                    del output_handle[group_name]
                sub_group = output_handle.create_group(group_name)
                recursive_dict_to_hdf5(sub_group, result)
    else:
        with multiprocessing.Pool(ncpu) as pool:
            for i, result in tqdm(
                enumerate(pool.imap(run_pipeline, prd_indices)),
                total=len(prd_indices),
                desc="Running manifold decomposition...",
            ):
                if return_output:
                    prd_data[i] = result
                if output_handle:
                    group_name = f"prd_{i}"
                    if output_handle.get(group_name):
                        del output_handle[group_name]
                    sub_group = output_handle.create_group(group_name)
                    recursive_dict_to_hdf5(sub_group, result)

    prds.guess_and_mark_anchors()
    prds.save()

    if return_output:
        return prd_data
