import numpy as np
from scipy.spatial.transform import Rotation

from ManifoldEM.util import eul_to_quat
from ManifoldEM.quaternion import quaternion_to_S2, collapse_to_half_space, pd_to_spider_euler, psi_ang, q_product
from euler_reference import (collapse_to_half_space_euler_angles, convert_euler_to_S2, qs_to_spider_euler_angles,
                             convert_S2_to_euler)


def test_eul_to_quat():
    """
    ManifoldEM raw quaternion convention.
    
    The convention is different from the one used in scipy.spatial.transform.Rotation.
    Converting between the two conventions required permuting indices and negating some values.
    """
    n_euler_angles = 3
    n_random_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))

    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi)
    q0, q1, q2, q3 = raw_qs

    intrinsic_zxz = 'ZXZ'
    q_ZXZ = Rotation.from_euler(intrinsic_zxz, euler_angles, degrees=False).as_quat()
    permuted_ZXZ = np.vstack([-q2, q1, -q3, q0]).T
    assert np.allclose(q_ZXZ, permuted_ZXZ, atol=atol)

    intrinsic_zyz = 'ZYZ'
    q_ZYZ = Rotation.from_euler(intrinsic_zyz, euler_angles, degrees=False).as_quat()
    permuted_ZYZ = np.vstack([-q1, -q2, -q3, q0]).T
    assert np.allclose(q_ZYZ, permuted_ZYZ, atol=atol)

    intrinsic_xyx = 'XYX'
    q_XYX = Rotation.from_euler(intrinsic_xyx, euler_angles, degrees=False).as_quat()
    permuted_XYX = np.vstack([-q3, -q2, q1, q0]).T
    assert np.allclose(q_XYX, permuted_XYX, atol=atol)

    intrinsic_xzx = 'XZX'
    q_XZX = Rotation.from_euler(intrinsic_xzx, euler_angles, degrees=False).as_quat()
    permuted_XZX = np.vstack([-q3, -q1, -q2, q0]).T
    assert np.allclose(q_XZX, permuted_XZX, atol=atol)

    intrinsic_xzx = 'YXY'
    q_YXY = Rotation.from_euler(intrinsic_xzx, euler_angles, degrees=False).as_quat()
    permuted_YXY = np.vstack([-q2, -q3, -q1, q0]).T
    assert np.allclose(q_YXY, permuted_YXY, atol=atol)

    # so called "spider" convention, w/ clockwise extrinsic rotations
    # this should give the legacy results directly from the original code
    extrinsic_zyz = 'zyz'
    q_zyz = Rotation.from_euler(extrinsic_zyz, -euler_angles, degrees=False).as_quat()
    permuted_zyz = np.vstack([q1, q2, q3, q0]).T
    assert np.allclose(q_zyz, permuted_zyz, atol=atol)


def test_quaternion_to_S2():
    n_euler_angles = 3
    n_random_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))

    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi)
    s2_mem = quaternion_to_S2(raw_qs)

    s2 = convert_euler_to_S2(euler_angles)

    assert np.allclose(s2, s2_mem, atol=atol)


def test_collapse_to_half_space():
    """
    Tests the collapse_to_half_space_euler_angles against the collapse_to_half_space function. 
    
    Can convert from Euler angles directly to S2, instead of going through quaternions.
    """
    n_euler_angles = 3
    n_random_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))
    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi)
    q_mem, is_mirrored_mem = collapse_to_half_space(raw_qs)
    s2_mem_collapsed = quaternion_to_S2(q_mem)

    s2, is_mirrored = collapse_to_half_space_euler_angles(euler_angles)

    assert np.allclose(s2, s2_mem_collapsed, atol=atol)
    assert np.allclose(is_mirrored_mem, is_mirrored)

    plane_vec = np.array([0.0, 1.0, 0.0])
    q_mem, is_mirrored_mem = collapse_to_half_space(raw_qs, plane_vec)
    s2_mem_collapsed = quaternion_to_S2(q_mem)

    s2, is_mirrored = collapse_to_half_space_euler_angles(euler_angles, plane_vec)
    assert np.allclose(s2, s2_mem_collapsed, atol=atol)
    assert np.allclose(is_mirrored_mem, is_mirrored)


def _spider_quat(phi, theta, psi):
    """The quaternion of Spider Euler angles, composed as the fit that pd_to_spider_euler replaced did."""
    def about_z(a):
        return np.array([np.cos(a / 2), 0.0, 0.0, -np.sin(a / 2)])

    def about_y(a):
        return np.array([np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0])

    return q_product(about_z(psi), q_product(about_y(theta), about_z(phi))).ravel()


def test_pd_to_spider_euler():
    np.random.seed(0)
    PDs = np.random.normal(size=(3, 1000))
    # the hemisphere edge x = 0, where the fit failed, both poles, and x < 0
    edge_cases = [[0, 0.8, 0.6], [1e-7, -0.8, 0.6], [-1e-7, 0.8, -0.6], [0, 0, 1], [0, 0, -1], [0, 1e-9, 1],
                  [-0.5, 0.5, 0.1]]
    PDs = np.hstack([PDs, np.array(edge_cases, dtype=float).T])
    PDs /= np.linalg.norm(PDs, axis=0)

    wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    for PD in PDs.T:
        phi, theta, psi = pd_to_spider_euler(PD)
        assert 0 <= theta <= np.pi
        assert psi == -phi

        # the Euler angles point the z-axis at PD
        assert np.allclose(Rotation.from_euler('ZYZ', [phi, theta, psi]).apply([0.0, 0.0, 1.0]), PD)

        # ... and give exactly the shortest rotation from z to PD (undefined at the south pole)
        Qr = np.array([1 + PD[2], PD[1], -PD[0], 0.0])
        if np.linalg.norm(Qr) < 1e-12:
            continue
        Qr /= np.linalg.norm(Qr)
        assert np.allclose(_spider_quat(phi, theta, psi), Qr)

        # scipy's conversion gives the same angles, or their other representation; at the north pole
        # (gimbal lock) only phi + psi is defined
        if theta > 1e-6:
            ref = qs_to_spider_euler_angles(Qr.reshape(4, 1)).ravel()
            other = ref + np.array([np.pi, 0.0, np.pi]) - np.array([0.0, 2 * ref[1], 0.0])
            ours = np.array([phi, theta, psi])
            assert np.allclose(wrap(ours - ref), 0, atol=1e-9) or np.allclose(wrap(ours - other), 0, atol=1e-9)


def test_psi_ang():
    n_euler_angles = 3
    n_random_samples = 1000

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))

    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi)
    s2_mem = quaternion_to_S2(raw_qs)
    projection_direction_euler_angles_mem = np.array([np.radians(psi_ang(s2)) for s2 in s2_mem.T]).T

    s2 = convert_euler_to_S2(euler_angles)
    projection_direction_euler_angles = convert_S2_to_euler(s2)

    assert np.allclose(projection_direction_euler_angles, projection_direction_euler_angles_mem, atol=1e-4)
