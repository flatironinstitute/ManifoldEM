import h5py
import logging
import mrcfile
import multiprocessing
import numpy as np
import sys

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
from ManifoldEM.DMembeddingII import op as diffusion_map_embedding
from ManifoldEM.fit_1D_open_manifold_3D import fit_1D_open_manifold_3D

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


def get_raw_images(image_stack_file: str, image_indices: NDArray[Shape["Any"], Int]):
    with mrcfile.mmap(image_stack_file, "r") as mrc:
        images = mrc.data

    raw_images = np.empty((len(image_indices), images.shape[1], images.shape[2]))
    for i, idx in enumerate(image_indices):
        raw_images[i] = images[idx]

    return raw_images


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
        distances, distances.shape[0], nlsa_tune, 60000
    )
    posPath1 = get_psiPath(psi, rad, 0)

    while len(posPath1) < nS:
        nS = len(posPath1)
        D1 = distances[posPath1][:, posPath1]
        lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = (
            diffusion_map_embedding(D1, D1.shape[0], nlsa_tune, 600000)
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


def _corr(a, b, n, m):
    A = a[:, n]
    B = b[:, m]
    A = A - np.mean(A)
    B = B - np.mean(B)
    try:
        co = np.dot(A, B) / (np.std(A) * np.std(B))
    except RuntimeError:
        raise RuntimeError("flat image")
    return co


def _diff_corr(a, b, maxval):
    return (
        _corr(a, b, 0, 0)
        + _corr(a, b, maxval, maxval)
        - (_corr(a, b, 0, maxval) + _corr(a, b, maxval, 0))
    )


def _NLSA(NLSAPar, DD, posPath, posPsi1, imgAll, msk2, CTF):
    num = NLSAPar["num"]
    ConOrder = NLSAPar["ConOrder"]
    k = NLSAPar["k"]
    tune = NLSAPar["tune"]
    nS = NLSAPar["nS"]
    psiTrunc = NLSAPar["psiTrunc"]

    ConD = np.zeros((num - ConOrder, num - ConOrder))
    for i in range(ConOrder):
        Ind = range(i, num - ConOrder + i)
        ConD += DD[Ind][:, Ind]

    # find the manifold mapping:
    lambdaC, psiC, _, mu, _, _, _, _ = diffusion_map_embedding(ConD, k, tune, 600000)

    lambdaC = lambdaC[lambdaC > 0]  ## lambdaC not used? REVIEW
    psiC1 = np.copy(psiC)
    # rearrange arrays
    IMG1 = imgAll[posPath[posPsi1], :, :]
    # Wiener filtering
    wiener_dom, CTF1 = get_wiener(CTF, posPath, posPsi1, ConOrder, num)

    dim = CTF.shape[1]
    ell = psiTrunc - 1
    N = psiC.shape[0]
    psiC = np.hstack((np.ones((N, 1)), psiC[:, 0:ell]))
    mu_psi = mu.reshape((-1, 1)) * psiC
    A = np.zeros((ConOrder * dim * dim, ell + 1), dtype="float64")
    tmp = np.zeros((dim * dim, num - ConOrder), dtype="float64")

    for ii in range(ConOrder):
        for i in range(num - ConOrder):
            ind1 = 0
            ind2 = dim * dim  # max(IMG1.shape)
            ind3 = ConOrder - ii + i - 1
            img = IMG1[ind3, :, :]
            img_f = fft2(img)  # .reshape(dim, dim)) T only for matlab
            CTF_i = CTF1[ind3, :, :]
            img_f_wiener = img_f * (CTF_i / wiener_dom[i, :, :])
            img = ifft2(img_f_wiener).real
            img = img * msk2  # April 2020
            tmp[ind1:ind2, i] = np.squeeze(img.T.reshape(-1, 1))

        mm = dim * dim  # max(IMG1.shape)
        ind4 = ii * mm
        ind5 = ind4 + mm
        A[ind4:ind5, :] = np.matmul(tmp, mu_psi)

    TF = np.isreal(A)
    if not TF.any():
        print("A is an imaginary matrix!")
        sys.exit()

    U, S, V = svdRF(A)
    VX = np.matmul(V.T, psiC.T)

    sdiag = np.diag(S)

    Npixel = dim * dim
    Topo_mean = np.zeros((Npixel, psiTrunc))
    for ii in range(psiTrunc):  # of topos considered
        # s = s + 1  needed?
        Topo = np.ones((Npixel, ConOrder)) * np.Inf

        for k in range(ConOrder):
            Topo[:, k] = U[k * Npixel : (k + 1) * Npixel, ii]
        Topo_mean[:, ii] = np.mean(Topo, axis=1)

    # unwrapping... REVIEW; allow user option to select from a list of chronos ([0,1,3]) to retain (i.e., not just i1, i2)
    i2 = 1
    i1 = 0

    ConImgT = np.zeros((max(U.shape), ell + 1), dtype="float64")
    for i in range(i1, i2 + 1):
        # %ConImgT = U(:,i) *(sdiag(i)* V(:,i)')*psiC';
        ConImgT = ConImgT + np.matmul(
            U[:, i].reshape(-1, 1), sdiag[i] * (V[:, i].reshape(1, -1))
        )

    recNum = ConOrder
    # tmp = np.zeros((Npixel,num-ConOrder),dtype='float64')
    IMGT = np.zeros((Npixel, nS - ConOrder - recNum), dtype="float64")
    for i in range(recNum):
        ind1 = i * Npixel
        ind2 = ind1 + Npixel
        tmp = np.matmul(ConImgT[ind1:ind2, :], psiC.T)
        for ii in range(num - 2 * ConOrder):
            ind3 = i + ii
            ttmp = IMGT[:, ii]
            ttmp = ttmp + tmp[:, ind3]
            IMGT[:, ii] = ttmp

    # normalize per frame so that mean=0 std=1, whole frame (this needs justif)
    for i in range(IMGT.shape[1]):
        ttmp = IMGT[:, i]
        try:
            ttmp = (ttmp - np.mean(ttmp)) / np.std(ttmp)
        except:
            print("flat image")
            exit(0)
        IMGT[:, i] = ttmp

    nSrecon = min(IMGT.shape)
    Drecon = L2_distance(IMGT, IMGT)
    k = nSrecon

    lamb, psirec, sigma, mu, logEps, logSumWij, popt, R_squared = (
        diffusion_map_embedding((Drecon**2), k, tune, 30)
    )

    lamb = lamb[lamb > 0]
    a, b, tau = fit_1D_open_manifold_3D(psirec)

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
    pos_path,
    senses,
    con_order_range,
    psi_trunc,
    first_pass_images=None,
):
    nS = len(pos_path)  # number of images in PD
    con_order = nS // con_order_range
    # if ConOrder is large, noise-free 2D frames expected w/ small range of conformations, \
    # while losing snapshots at edges

    pos_path = np.squeeze(pos_path)
    distances = distances[pos_path][:, pos_path]

    res = []
    for psinum in psi_list:  # for each reaction coordinates do the following:
        if psinum == -1:
            continue

        # e.g., shape=(numPDs,): reordering image indices along each diff map coord
        psi_sorted_ind = np.argsort(psi[:, psinum])
        DD = distances[psi_sorted_ind]
        # distance matrix with indices of images re-arranged along current diffusion map coordinate
        DD = DD[:, psi_sorted_ind]
        num = DD.shape[1]  # number of images in PD (duplicate of nS?)
        k = num - con_order

        NLSAPar = dict(
            num=num,
            ConOrder=con_order,
            k=k,
            tune=params.nlsa_tune,
            nS=nS,
            save=False,
            psiTrunc=psi_trunc,
        )
        IMGT, Topo_mean, psirec, psiC1, sdiag, VX, mu, tau = _NLSA(
            NLSAPar, DD, pos_path, psi_sorted_ind, images, mask, CTF
        )

        n_s_recon = min(IMGT.shape)
        numclass = min(params.states_per_coord, n_s_recon // 2)

        tau = (tau - min(tau)) / (max(tau) - min(tau))
        i1 = 0
        i2 = IMGT.shape[0]

        IMG1 = np.zeros((i2, numclass), dtype=np.float16)
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

            IMG1[i1:i2, i] = IMGT[:, tauind[0]]
            tauinds[i] = tauind[0]
        if first_pass_images is not None:  # second pass for energy_landscape
            #  adjust tau by comparing the first pass output images
            dc = _diff_corr(IMG1, first_pass_images, numclass - 1)
            if (senses[0] == -1 and dc > 0) or senses[0] == 1 and dc < 0:
                tau = 1 - tau

            res.append(
                dict(
                    IMG1=IMG1.astype(np.float16),
                    IMGT=IMGT.astype(np.float16),
                    posPath=pos_path,
                    PosPsi1=psi_sorted_ind,
                    psirec=psirec,
                    tau=tau,
                    psiC1=psiC1,
                    mu=mu,
                    VX=VX,
                    sdiag=sdiag,
                    Topo_mean=Topo_mean.astype(np.float16),
                    tauinds=tauinds,
                )
            )
        else:  # first pass
            res.append(
                dict(
                    IMG1=IMG1.astype(np.float16),
                    psirec=psirec,
                    tau=tau,
                    psiC1=psiC1,
                    mu=mu,
                    VX=VX,
                    sdiag=sdiag,
                    Topo_mean=Topo_mean.astype(np.float16),
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
        True,
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
        np.ones(params.num_psi),
        params.con_order_range,
        params.num_psi_truncated,
    )

    image_data = dict(filter=image_filter, mask=image_mask, rotations=image_rotations, avg_orientation=avg_orientation)
    res = {
        "image_data": image_data,
        "distances": distances_data,
        "embedding_data": embed_data,
        "nlsa_data": nlsa_data,
    }

    return res


def recursive_dict_to_hdf5(group: h5py.Group, data: dict[str, Any]):
    """Recursively writes a dictionary to an HDF5 group.
    Data must be something convertible to an HDF5 dataset (e.g. a numpy array).
    Adds groups based on dictionary keys. Existing data will be overwritten.

    Params
    ------
    group: h5py.Group
        Group (including root h5py.File object) to write
    data: dict[str, Any]
        Any data you want to dump
    """
    for key, item in data.items():
        if isinstance(item, dict):
            sub_group = group.create_group(key)
            recursive_dict_to_hdf5(sub_group, item)
        elif isinstance(item, list):
            for i, sub_item in enumerate(item):
                if isinstance(sub_item, dict):
                    sub_group = group.create_group(f"{key}_{i}")
                    recursive_dict_to_hdf5(sub_group, sub_item)
                else:
                    group.create_dataset(f"{key}_{i}", data=sub_item)
        else:
            group.create_dataset(key, data=item)


def prd_analysis(
    project_file: None | str,
    prd_indices: None | Iterable[int] = None,
    return_output: bool = True,
    output_handle: None | h5py.Group = None,
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

    if prd_indices is None:
        prd_indices = list(range(params.prd_n_active))
    else:
        prd_indices = list(prd_indices)

    tqdm = get_tqdm()
    prd_data: dict[int, Any] = {}

    with multiprocessing.Pool(processes=params.ncpu) as pool:
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

    if return_output:
        return prd_data
