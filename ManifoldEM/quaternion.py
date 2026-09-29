import copy
import numpy as np

from typing import Any
from nptyping import NDArray, Shape, Bool, Float
"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
"""


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


def pd_to_spider_euler(PD: NDArray[Shape["3"], Float]) -> tuple[float, float, float]:
    """
    Euler angles, in radians, of the shortest rotation that takes the z-axis to a projection direction.

    The rotation is the quaternion [1 + z, y, -x, 0] (normalized) for PD = (x, y, z), written in the
    Spider ZYZ convention. Of its two Euler representations, this returns the one with theta in
    [0, pi]: phi = atan2(y, x), theta = arccos(z), psi = -phi. At the poles, phi = psi = 0.

    Parameters
    ----------
    PD : np.ndarray
        The projection direction, a 3-vector; it is normalized here.

    Returns
    -------
    tuple[float, float, float]
        (phi, theta, psi)

    Notes
    -----
    This replaces a least-squares fit (`q2Spider`) that started at phi = theta = psi = 0, which is
    gimbal lock. There, the fit sometimes stopped at its start point, depending on the state of the
    process, and it always did within about 1e-5 of x = 0. Where it converged, it agreed with this
    to about 1e-10 for x > 0, and gave the other representation, (phi + pi, -theta, psi + pi), for x < 0.
    """
    x, y, z = np.asarray(PD, dtype=float).ravel() / np.linalg.norm(PD)
    phi = float(np.arctan2(y, x))
    theta = float(np.arctan2(np.hypot(x, y), z))
    return (phi, theta, -phi)


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
    - This function reshapes 1-dimensional input arrays to 2D for consistent processing.
    """
    # if 1-dim vector
    if len(q.shape) < 2:
        q = q.reshape(-1, 1)
    if len(s.shape) < 2:
        s = s.reshape(-1, 1)
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
    phi, theta, _ = pd_to_spider_euler(PD)

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
