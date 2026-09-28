"""Reference implementations used as test oracles.

scipy-based Euler angle / S2 conversions, moved here from ManifoldEM.quaternion because only the
tests use them: they check the package's own routines (q2Spider, collapse_to_half_space,
core.get_euler_from_PD, ...) against these independent implementations.
"""
from typing import Any, Tuple

import numpy as np
from nptyping import NDArray, Shape, Bool, Float
from scipy.spatial.transform import Rotation


def convert_euler_to_S2(euler_angles: NDArray[Shape["Any,3"], Float]) -> NDArray[Shape["Any,3"], Float]:
    """
    Converts an array of euler angles in ZXZ representation to an array of unit vectors on S2

    Parameters
    ----------
    euler_angles : np.ndarray
        An Nx3 numpy array representing the Euler angles in radians.

    Returns
    -------
    np.ndarray
        A Nx3 array of unit vectors corresponding to the input Euler angles.
    """
    rotation = Rotation.from_euler('ZXZ', euler_angles, degrees=False).as_matrix()
    rxz, ryz, rzz = rotation[:, :, -1].T
    return np.stack([-ryz, rxz, rzz], axis=1).T


def collapse_to_half_space_euler_angles(
    euler_angles: NDArray[Shape["Any,3"], Float],
    plane_vec: NDArray[Shape["3"], Float] = np.array([1.0, 0.0, 0.0]),
) -> Tuple[NDArray[Shape["Any,3"], Float], NDArray[Shape["Any"], Bool]]:
    """
    Converts an array of euler angles in ZXZ representation to an array of unit vectors on half of S2,
    with the division based on a plane vector. Points on the opposite side of the plane vector are mirrored,
    and a boolean array is returned to indicate the mirrored points.

    Parameters
    ----------
    euler_angles : np.ndarray
        An Nx3 numpy array representing the Euler angles in radians.
    plane_vec : np.ndarray
        A 3-element numpy array representing the plane vector that divides S2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the unit vectors (Nx3) and a boolean array (N) indicating mirrored points.
    """

    s2 = convert_euler_to_S2(euler_angles)
    is_mirrored = np.dot(s2.T, plane_vec) < 0.0
    s2[:, is_mirrored] = -s2[:, is_mirrored]
    return s2, is_mirrored


def qs_to_spider_euler_angles(raw_qs: NDArray[Shape["4,Any"], Float]) -> NDArray[Shape["3,Any"], Float]:
    """
    Convert quaternions to Euler angles in the Spider convention.

    Parameters
    ----------
    raw_qs : np.ndarray
       A 4xN numpy array representing the quaternions in [R,I,I,I] format

    Returns
    -------
    np.ndarray
        3xN array of Euler angles in the "spider" convention
    """
    # https://www.ccpem.ac.uk/user_help/rotation_conventions.php
    # Claims anti-clockwise, though clockwise gives the same results as legacy q2Spider routine...
    # "ZYZ anti-clockwise. (α,β,γ) is (ψ,θ,φ), note reversal because Spider defines its angles w.r.t external axes."
    # so negate euler angles for clockwise, and use lower case for extrinsic in scipy transforms
    #
    # python 3.9 doesn't have a scipy (1.14+) that supports the convenient scalar_first argument, so we need to permute to
    # get the scalar in the last position
    qs_scalar_last = np.vstack((raw_qs[1:4, :], raw_qs[0, :]))
    return -Rotation.from_quat(qs_scalar_last.T).as_euler('zyz').T


def alternate_euler_convention(euler_angles: NDArray[Shape["3,Any"], Float]) -> NDArray[Shape["3,Any"], Float]:
    """
    FIXME -- what's going on here?
    Maps Euler angles in the ??? convention to the ??? convention

    Parameters
    ----------
    euler_angles_deg : np.ndarray
        3xN array of Euler angles in the ??? convention

    Returns
    -------
    np.ndarray
        3xN array of Euler angles in the ??? convention
    """
    euler_angles_alternate = np.mod(((euler_angles.T - np.array([np.pi, 0, np.pi])) * np.array([1, -1, 1])).T,
                                    2 * np.pi)
    return euler_angles_alternate


def convert_S2_to_euler(s2):
    """
    FIXME -- what's going on here?
    Maps unit vectors representing positions on in S2 to Euler angles in the ??? convention.

    Parameters
    ----------
    s2 : np.ndarray
        3xN of unit vectors

    Returns
    -------
    np.ndarray
        3xN array of Euler angles in the ??? convention
    """
    sx, sy, sz = s2
    x = np.arccos(sz)  # Polar angle
    z1 = np.arctan2(sy, sx)  # First Euler angle
    angles = np.mod(np.array([z1, x, np.zeros_like(x)]), 2 * np.pi)
    angles_alternate = np.mod(alternate_euler_convention(angles), 2 * np.pi)
    convention_psi = 0.0
    angles_alternate[-1, :] = convention_psi

    mask = sx < 0
    angles[:, mask] = angles_alternate[:, mask]

    return angles
