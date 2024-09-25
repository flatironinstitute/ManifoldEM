import mrcfile
import numpy as np
import os
import sys
import pickle
import traceback

from typing import Any
from nptyping import NDArray, Shape, Float64

from scipy.ndimage import rotate

from ManifoldEM.params import params
from ManifoldEM.quaternion import q_product

"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
"""


class NullEmitter:
    """
    A class that provides a no-operation (no-op) implementation of an emmitter for progress tracking
    for tqdm/QT progress bars.

    This class is designed to be used in contexts where an emitter is required by the interface,
    but no actual emitting action is desired.
    """

    def emit(self, percent):
        """
        No-op method that does nothing.

        Parameters
        ----------
        percent : int
            The percentage of the progress that has been completed. Ignored.

        Returns
        -------
        None
        """
        pass


def get_tqdm():
    """
    Get jupyter version of tqdm function if in jupyter notebook, otherwise returns the terminal variant.
    Exploits the fact that the get_ipython function is only defined in jupyter/ipython processes and that its
    class name is 'ZMQInteractiveShell' in jupyter.

    Returns
    -------
    tqdm
        The tqdm function to use for progress tracking.
    """
    try:
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm

            return tqdm
    except NameError:  #  get_ipython not defined on base python...
        pass
    from tqdm import tqdm

    return tqdm


def remote_runner(hostname, cmd, progress_callback):
    from fabric import Connection

    with Connection(hostname, inline_ssh_env=True) as c:
        c.config.run.env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("PATH", "PYTHON", "VIRTUAL_ENV"))
        }
        param_file = os.path.join(os.getcwd(), f"params_{params.project_name}.toml")
        c.run(f"{cmd} {param_file}")
        progress_callback.emit(100)


def is_valid_host(hostname):
    from fabric import Connection

    try:
        with Connection(hostname) as c:
            c.run("true")
    except:
        return False

    return True


def get_image_width_from_stack(stack_file: str):
    """
    Determines the width of images in a file stack (pixels per row). Images assumed square.

    Parameters
    ----------
    stack_file :  str
        The path to the stack file. Works on .mrc and .mrcs files.

    Returns
    -------
    int
        The number of pixels per row of the images contained in the stack file.

    Raises
    ------
    ValueError
        If the file is not an MRC or MRCs file.

    """
    img_width = 0
    if stack_file.endswith(".mrcs") or stack_file.endswith(".mrc"):
        mrc = mrcfile.mmap(params.img_stack_file, mode="r")
        if not mrc.is_image_stack():
            mrc.close()
            mrc = mrcfile.mmap(params.img_stack_file, mode="r+")
            mrc.set_image_stack()

        img_width = mrc.data[0].shape[0]
        mrc.close()
    else:
        raise ValueError("Particles must be in mrc or mrcs format.")

    return img_width


def calc_shannon(res: float, dia: float) -> float:
    return res / dia


def calc_ang_width(aperture: int, resolution: float, diameter: float) -> float:
    return min(aperture * resolution / diameter, 4 * np.sqrt(4 * np.pi))


def debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook

    try:
        from ipdb import set_trace
    except:
        from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def debug_print(msg: str = ""):
    """
    Prints a debug message along with the caller's stack trace.

    Parameters
    ----------
    msg : str
        The debug message to print. If empty, only the stack trace is printed.

    Returns
    -------
    None
    """

    if msg:
        print(msg)
    stack = traceback.format_stack()

    print(stack[-2].split("\n")[0])


def hist_match(source, template):  # by ali_m
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def histeq(src, thist):  # by Zagurskin; does not work well,
    nbr_bins = len(thist)
    bins = np.linspace(0, 1, nbr_bins + 1)
    # hist, bins = np.histogram(src.flatten(), nbr_bins, normed=True)
    hist, bb = np.histogram(src.flatten(), bins)  # nbr_bins, normed=True)

    cdfsrc = hist.cumsum()  # cumulative distribution function
    cdfsrc = (nbr_bins * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

    cdftint = thist.cumsum()  # cumulative distribution function
    cdftint = (nbr_bins * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

    h2 = np.interp(src.flatten(), bins[:-1], cdfsrc)
    h3 = np.interp(h2, cdftint, bins[:-1])

    return h3


def eul_to_quat(phi, theta, psi, flip=True):
    """
    Converts Euler angles to quaternions.

    Parameters
    ----------
    phi : np.ndarray
        Array of rotations around the z-axis.
    theta : np.ndarray
        Array of rotations around the y-axis.
    psi : np.ndarray
        Array of rotations around the x-axis.
    flip : bool, default=True
        If True, flips the sign of the psi component. Defaults to True.

    Returns
    -------
    np.ndarray
        An array of quaternions representing the rotations. Each quaternion is
        represented as a column in a 4xN array, where N is the number of sets of Euler angles.

    Notes
    -----
    - The function constructs individual quaternions for rotations around the z, y, and x axes
      (qz, qy, qzs respectively) and then combines them through quaternion multiplication to
      obtain the final quaternion representing the combined rotation.
    - The `flip` parameter can be used to adjust for different conventions in the definition
      of rotations.
    """

    zros = np.zeros(phi.shape[0])
    qz = np.vstack((np.cos(phi / 2), zros, zros, -np.sin(phi / 2)))
    qy = np.vstack((np.cos(theta / 2), zros, -np.sin(theta / 2), zros))
    sp = -np.sin(psi / 2) if flip else np.sin(psi / 2)
    qzs = np.vstack((np.cos(psi / 2), zros, zros, sp))
    q = q_product(qzs, q_product(qy, qz))
    return q


def augment(q: NDArray[Shape["4,*"], Float64]):
    """
    Augments a set of quaternions by adding their conjugates to the set.

    Parameters
    ----------
    q : np.ndarray
        A numpy array of shape (4, N) representing a set of quaternions,
        where N is the number of quaternions.

    Returns
    -------
    ndarray
        An augmented numpy array of shape (4, 2N) containing the original
        set of quaternions followed by their conjugates.

    Raises
    ------
    AssertionError
        If the input array does not have the correct shape (i.e., does not
        have 4 rows representing quaternions).

    Notes
    -----
    - The conjugate of a quaternion [q0, q1, q2, q3] is defined as [-q1, q0, -q3, -q2].
    - This function is useful for operations that require both a quaternion and its inverse,
      such as applying rotations and their reversals.
    """

    try:
        assert q.shape[0] > 3
    except AssertionError:
        _logger.error("subroutine augment: q has wrong dimensions")
        _logger.exception("subroutine augment: q has wrong diemnsions")
        raise
        sys.exit(1)

    qc = np.vstack((-q[1, :], q[0, :], -q[3, :], q[2, :]))
    q = np.hstack((q, qc))

    return q


def make_indeces(inputGCs):
    with open(inputGCs, "rb") as f:
        param = pickle.load(f)
    f.close()

    GCnum = len(param["CGtot"])
    prDs = len(param["CGtot"][0])

    x1 = np.tile(range(prDs), (1, GCnum))
    x2 = np.array([])
    for i in range(GCnum):
        x2 = np.append(x2, np.tile(i, (1, prDs)))
    xAll = np.vstack((x1, x2)).astype(int)
    xSelect = range(xAll.shape[1])

    return xAll, xSelect


def interv(s):
    # return np.arange(-s/2,s/2)
    if s % 2 == 0:
        a = -s / 2
        b = s / 2 - 1
    else:
        a = -(s - 1) / 2
        b = (s - 1) / 2

    return np.linspace(a, b, s)


def filter_fourier(inp, sigma):
    # filter Gauss
    nPix1 = inp.shape[1]
    nPix2 = inp.shape[0]
    X, Y = np.meshgrid(interv(nPix1), interv(nPix2))
    Rgrid = nPix2 / 2.0
    Q = (1 / Rgrid) * np.sqrt(X**2 + Y**2)  # Q in units of Nyquist frequency

    N = 4
    G = np.sqrt(1.0 / (1 + (Q / sigma) ** (2 * N)))  # ButterWorth

    # Filter images in Fourier space
    G = np.fft.ifftshift(G)
    inp = np.real(np.fft.ifft2(G * np.fft.fft2(inp)))

    return inp


def create_proportional_grid(N: int) -> NDArray[Shape["*,*"], Float64]:
    """
    Creates an NxN grid centered around (0, 0).

    The function generates an NxN grid where each point's value is proportional to its distance from the center,
    normalized by the grid size. This can be used for generating spatial frequency grids or other applications
    where a centered grid is required.

    Parameters
    ----------
    N : int
        The linear size of the grid (i.e. width).

    Returns
    -------
    ndarray
        An `NxN` NumPy array representing the grid
    """
    a = np.arange(N) - N // 2
    X, Y = np.meshgrid(a, a)

    return 2 * np.sqrt(X**2 + Y**2) / N


def ctemh_cryoFrank(
    k: NDArray[Shape["Any,Any"], Float64],
    spherical_aberration: float,
    defocus: float,
    electron_energy: float,
    gauss_env_halfwidth: float,
    amplitude_contrast_ratio: float,
):
    """
    Calculates the contrast transfer function (CTF) for cryo-EM imaging.

    Parameters
    ----------
    k : ndarray
        A 2D array of spatial frequencies.
    spherical_aberration : float
        Spherical aberration (Cs) in mm.
    defocus : float
        Defocus in Angstroms. A positive value indicates underfocus.
    electron_energy : float
        Electron energy in keV.
    gauss_env_halfwidth : float
        Half-width of the Gaussian envelope in A^-2.
    amplitude_contrast_ratio : float
        Amplitude contrast ratio obtained from the alignment file.

    Returns
    -------
    ndarray
        A 2D array representing the CTF of shape `k`.

    Notes
    -----
    - we assume |k| = s
    - from Kirkland, adapted for cryo (EMAN1) by P. Schwander
    - Here, the damping envelope is characterized by a single parameter B (gauss_env)
    - see J. Frank

    Copyright (c) UWM, Peter Schwander 2010 MATLAB version
    Copyright (c) Columbia University Hstau Liao 2018 (python version)
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


def get_CTFs(
    width: int,
    defocus: NDArray[Shape["Any"], Float64],
    spherical_aberration: float,
    electron_energy: float,
    gauss_env_halfwidth: float,
    amplitude_contrast_ratio: float,
) -> NDArray[Shape["Any,Any,Any"], Float64]:
    """
    Calculates the contrast transfer function (CTF) for cryo-EM imaging.

    Parameters
    ----------
    width : int
        The width of the generated image.
    defocus : ndarray
        The defocus values of interest.
    spherical_aberration : float
        Spherical aberration (Cs) in mm.
    defocus : float
        Defocus in Angstroms. A positive value indicates underfocus.
    electron_energy : float
        Electron energy in keV.
    gauss_env_halfwidth : float
        Half-width of the Gaussian envelope in A^-2.
    amplitude_contrast_ratio : float
        Amplitude contrast ratio obtained from the alignment file.

    Returns
    -------
    ndarray
        A 3D array representing the CTF of shape `(len(defocus), width, width)`.
    """

    k = create_proportional_grid(width) / (2 * width)
    ctf = np.empty((len(defocus), width, width))
    for i, df in enumerate(defocus):
        ctf[i] = np.fft.ifftshift(
            ctemh_cryoFrank(
                k,
                spherical_aberration,
                df,
                electron_energy,
                gauss_env_halfwidth,
                amplitude_contrast_ratio,
            )
        )

    return ctf


def rotate_fill(img: NDArray[Shape["*,*"], Float64], angle: float) -> NDArray[Shape["*,*"], Float64]:
    """
    Rotates an image by a given angle and fills the output image by repeating the input image.

    Parameters
    ----------
    img : ndarray
        The input image as a 2D NumPy array.
    angle :
        The rotation angle in degrees.

    Returns
    -------
    ndarray
        The rotated image as a 2D NumPy array.
    """
    return rotate(img, angle, reshape=False, mode='grid-wrap')
