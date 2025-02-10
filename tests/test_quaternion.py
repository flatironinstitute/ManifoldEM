import numpy as np
from scipy.spatial.transform import Rotation

from ManifoldEM.util import eul_to_quat
from ManifoldEM.quaternion import (quaternion_to_S2, collapse_to_half_space, collapse_to_half_space_euler_angles,
                                   convert_euler_to_S2, q2Spider, qs_to_spider_euler_angles, psi_ang,
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
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
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


def test_quaternion_to_S2():
    n_euler_angles = 3
    n_random_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
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
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
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


def test_q2Spider():
    n_euler_angles = 3
    n_random_samples = 1000

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))
    euler_angles[:, 1] = 0.5 * (euler_angles[:, 1] + np.pi)

    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi, flip=True)

    euler_angles_mem = np.array([q2Spider(raw_q) for raw_q in raw_qs.T]).T
    euler_angles_spider = qs_to_spider_euler_angles(raw_qs)

    # Euler angles are not unique, so check that they produce the same S2 points
    s2_prod = quaternion_to_S2(raw_qs).T
    s2_prod_scipy = Rotation.from_euler('ZYZ', euler_angles_mem.T).apply(np.array([0.0, 0.0, 1.0]))
    s2_spider = Rotation.from_euler('ZYZ', euler_angles_spider.T).apply(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(s2_prod, s2_prod_scipy, atol=1e-4)
    assert np.allclose(s2_prod, s2_spider, atol=1e-4)

    # also check quaternions, which are unique up to a sign
    back_qs_scalar_last = Rotation.from_euler('zyz', euler_angles_spider.T).as_quat().T
    back_qs = np.vstack([back_qs_scalar_last[3], back_qs_scalar_last[0:3]])
    t1 = np.isclose(back_qs, raw_qs)
    t2 = np.isclose(back_qs, -raw_qs)
    assert np.logical_xor(t1, t2).all()


def test_psi_ang():
    n_euler_anlges = 3
    n_ransom_samples = 1000

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    s2_mem = quaternion_to_S2(raw_qs)
    projection_direction_euler_angles_mem = np.array([np.radians(psi_ang(s2)) for s2 in s2_mem.T]).T

    s2 = convert_euler_to_S2(euler_angles)
    projection_direction_euler_angles = convert_S2_to_euler(s2)

    assert np.allclose(projection_direction_euler_angles, projection_direction_euler_angles_mem, atol=1e-4)
