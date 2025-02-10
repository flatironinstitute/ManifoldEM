import copy
import numba
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation

from typing import Any, Tuple
from nptyping import NDArray, Shape, Bool, Float
"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
"""


@numba.jit(nopython=True)
def _q_product_single(q1, q2):
    """
    Computes the product of two quaternions using Numba for JIT compilation.

    Parameters
    ----------
    q1 : np.ndarray
        The first quaternion as a NumPy array.
    q2 : np.ndarray
        The second quaternion as a NumPy array.

    Returns
    -------
    np.ndarray
        The product of the two quaternions as a new NumPy array.

    Notes
    -----
    This function is optimized for performance with Numba's nopython mode, ensuring
    that the computation is compiled to machine code for faster execution. Designed
    for use with the `q2Spider` function.
    """
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ])


@numba.jit(nopython=True)
def _optfunc(a, q):
    """
    Optimized function to compute a quaternion operation, intended for use for the optimizer in `q2Spider`.

    Parameters
    ----------
    a : np.ndarray
        A NumPy array representing angles.
    q : np.ndarray
        A target quaternion for the operation.

    Returns
    -------
    np.ndarray
        The result of the quaternion operation as a NumPy array.

    This function constructs three quaternions based on the input angles 'a', performs
    quaternion multiplication, and compares the result to the target quaternion 'q'.
    It is optimized with Numba's nopython mode for faster execution.
    """
    b = 0.5 * a
    q1 = np.array([np.cos(b[0]), 0., 0., -np.sin(b[0])])
    q2 = np.array([np.cos(b[1]), 0., -np.sin(b[1]), 0.])
    q3 = np.array([np.cos(b[2]), 0., 0., -np.sin(b[2])])
    return q - _q_product_single(q3, _q_product_single(q2, q1))


def quaternion_to_S2(q: NDArray[Shape["4,Any"], Float]) -> NDArray[Shape["3,Any"], Float]:
    """
    Converts a set of quaternions to points on the 2-sphere (S2) in three-dimensional space. The
    function effectively rotates the z unit vector by each quaternion to a new unit vector on the unit
    sphere. The rotation about the final axis (a.k.a. psi rotation) is ignored.

    Parameters
    ----------
    q : np.ndarray
        A numpy array of shape (4, N) where each column represents a quaternion. The quaternion
        is expected to be in the form [q0, q1, q2, q3] with q0 as the real part and q1, q2, q3 as
        the imaginary parts.

    Returns
    -------
    np.ndarray
        A numpy array of shape (3, N) where each column represents the 3D coordinates [x, y, z] of
        a point on the unit sphere (S2). These coordinates correspond to the orientation represented
        by the input quaternions.

    Notes
    -----
    - The conversion formula used in this function maps the quaternion's rotation to a point on
      the sphere by selecting specific combinations of the quaternion components. This mapping
      focuses on the spatial orientation and ignores the final rotation around the axis, which
      is not represented on the 2-sphere.
    - The function is useful in contexts where the orientation needs to be visualized or analyzed
      without the complexity of handling full quaternion rotations, such as in computer graphics,
      robotics, and orientation tracking.
    """
    # TODO: Understand how this magically gets rid of final psi rotation
    S2 = 2 * np.vstack(
        (q[1, :] * q[3, :] - q[0, :] * q[2, :], q[0, :] * q[1, :] + q[2, :] * q[3, :], q[0, :]**2 + q[3, :]**2 - 0.5))
    return S2


def collapse_to_half_space(
    q: NDArray[Shape["4,Any"], Float],
    plane_vec: NDArray[Shape["3"], Float] = np.array([1.0, 0.0, 0.0]),
) -> tuple[NDArray[Shape["4,Any"], Float], NDArray[Shape["Any"], Bool]]:
    """
    Converts an array of quaternions in the Manifold [R,I,I,I] representation to an array of unit vectors on half of S2,
    with the division based on a plane vector. Points on the opposite side of the plane vector are mirrored,
    and a boolean array is returned to indicate the mirrored points.

    Parameters
    ----------
    q : np.ndarray
        An Nx4 numpy array representing the quaternions.
    plane_vec : np.ndarray
        A 3-element numpy array representing the plane vector that divides S2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the unit vectors (Nx3) and a boolean array (N) indicating mirrored points.
    """

    n_particles = q.shape[1]
    q = copy.copy(q)
    S2 = quaternion_to_S2(q)
    is_mirrored = np.zeros(shape=(n_particles), dtype=bool)
    for i in range(n_particles):
        if np.dot(S2[:, i], plane_vec) < 0.0:
            q[:, i] = np.array([-q[1, i], q[0, i], -q[3, i], q[2, i]])
            is_mirrored[i] = True

    return (q, is_mirrored)


def q2Spider(q):
    """
    Converts a quaternion to a rotation sequence in the Spider 3D convention.

    Parameters
    ----------
    q : np.ndarray
        A quaternion represented as a 1D NumPy array of shape [4].

    Returns
    -------
    tuple[float]
        A tuple (phi, theta, psi) representing the rotation sequence in radians.

    The function normalizes the input quaternion to ensure it represents a valid rotation.
    It then uses an optimization process to find the Euler angles (phi, theta, psi) that correspond
    to the given quaternion. The Spider 3D convention is used, which is common in cryo-EM for
    representing 3D orientations.

    Note:
    - This function assumes the input quaternion is a unit quaternion.
    - The optimization process uses the Levenberg-Marquardt algorithm through scipy's least_squares method.
    - Copyright (c) Columbia University Hstau Liao 2018 (python version)
    """
    # assert unit quaternion
    q = q / np.linalg.norm(q)

    def dev1(a):
        return _optfunc(a, q)

    lb = -np.inf
    ub = np.inf
    tol = 1e-12

    a0 = np.array([0, 0, 0])
    res = optimize.least_squares(dev1, a0, bounds=(lb, ub), ftol=tol, method='lm')
    a = res.x

    phi = a[0]
    theta = a[1]
    psi = a[2]

    return (phi, theta, psi)


def q_product(q, s):
    """
    Calculates the quaternion product of two quaternions or arrays of quaternions.

    Parameters
    ----------
    q : np.ndarray
        A 4xN or 4x1 numpy array representing the first quaternion(s),
        where N is the number of quaternions.
    s : np.ndarray
        A 4xN or 4x1 numpy array representing the second quaternion(s),
        matching the dimensions of q.

    Returns
    -------
    np.ndarray
        The quaternion product, in the same format as the inputs (4xN or 4x1).

    Notes
    -----
    - Quaternions are represented as [q0, q1, q2, q3], where q0 is the scalar part,
      and [q1, q2, q3] represent the vector part.
    - The function supports broadcasting, allowing for the multiplication of a single
      quaternion with an array of quaternions and vice versa.
    - Quaternion multiplication is not commutative; the order of operands affects the result.
    - This function reshapes 1-dimensional input arrays to 2D for consistent processing
      and uses assertions to ensure proper input dimensions.
    """
    # if 1-dim vector
    if len(q.shape) < 2:
        q = q.reshape(-1, 1)
    if len(s.shape) < 2:
        s = s.reshape(-1, 1)
    try:
        assert (q.shape[0] > 3 and s.shape[0] > 3)
    except AssertionError:
        print('subroutine qMult_bsx: some vector have less than 4 elements')
    q0 = q[0, :]
    qv = q[1:4, :]
    s0 = s[0, :]
    sv = s[1:4, :]

    c = np.vstack((qv[1, :] * sv[2, :] - qv[2, :] * sv[1, :], qv[2, :] * sv[0, :] - qv[0, :] * sv[2, :],
                   qv[0, :] * sv[1, :] - qv[1, :] * sv[0, :]))

    p = np.vstack((q0 * s0 - np.sum(qv * sv, axis=0), q0 * sv + s0 * qv + c))
    return p


def psi_ang(PD: NDArray[Shape["3,1"], Float]) -> tuple[float, float, float]:
    """
    Converts a projection direction to Euler angles.
    
    Parameters
    ----------
    PD : np.ndarray
        A 3x1 numpy array representing the unit projection direction.

    Returns
    -------
    tuple[float,float,float]
        Euler angles: phi, theta, psi
    """
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0])
    Qr = Qr / np.sqrt(np.sum(Qr**2))
    phi, theta, psi = q2Spider(Qr)

    phi = np.mod(phi, 2 * np.pi) * (180 / np.pi)
    theta = np.mod(theta, 2 * np.pi) * (180 / np.pi)
    psi = 0.0  # degenerate, set to zero for convention

    return (phi, theta, psi)


def calc_avg_pd(q, nS):
    # Calculate average projection directions (from matlab code)
    """PDs = 2*[q(2,:).*q(4,:) - q(1,:).*q(3,:);...
    q(1,:).*q(2,:) + q(3,:).*q(4,:); ...
    q(1,:).^2 + q(4,:).^2 - ones(1,nS)/2 ];
    """
    PDs = 2 * np.vstack((q[1, :] * q[3, :] - q[0, :] * q[2, :], q[0, :] * q[1, :] + q[2, :] * q[3, :],
                         q[0, :]**2 + q[3, :]**2 - np.ones((1, nS)) / 2.0))

    return PDs


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
    # "ZYZ anti-clockwise. (α,β,γ) is (ψ,θ,φ), note reversal because Spider defines its angles w.r.t external axes."
    # so negate euler angles for counter-clockwise, and use lower case for extrinsic in scipy transforms
    return -Rotation.from_quat(raw_qs.T, scalar_first=True).as_euler('zyz').T


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
