import numpy as np
from scipy.spatial.transform import Rotation

from ManifoldEM.util import eul_to_quat
from ManifoldEM.quaternion import quaternion_to_S2, collapse_to_half_space, collapse_to_half_space_euler_angles, convert_euler_to_S2, q2Spider, qs_to_spider, psi_ang, convert_S2_to_euler

def test_eul_to_quat():
    ''' ManifoldEM raw quaternion convention.
    
    The convention is different from the one used in scipy.spatial.transform.Rotation.
    Converting between the two conventions required permuting indices and negating some values.
    '''
    n_euler_anlges = 3
    n_ransom_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    q0, q1, q2, q3 = raw_qs

    intrinsic_zxz = 'ZXZ'
    q_ZXZ = Rotation.from_euler(intrinsic_zxz, euler_angles, degrees=False).as_quat()
    permuted_ZXZ = np.vstack([-q2, q1, -q3, q0,]).T
    assert np.allclose(q_ZXZ,permuted_ZXZ,atol=atol)

    intrinsic_zyz = 'ZYZ'
    q_ZYZ = Rotation.from_euler(intrinsic_zyz, euler_angles, degrees=False).as_quat()
    permuted_ZYZ = np.vstack([-q1, -q2, -q3, q0,]).T
    assert np.allclose(q_ZYZ,permuted_ZYZ,atol=atol)

    intrinsic_xyx = 'XYX'
    q_XYX = Rotation.from_euler(intrinsic_xyx, euler_angles, degrees=False).as_quat()
    permuted_XYX = np.vstack([-q3,-q2,q1,q0,]).T
    assert np.allclose(q_XYX,permuted_XYX,atol=atol)

    intrinsic_xzx = 'XZX'
    q_XZX = Rotation.from_euler(intrinsic_xzx, euler_angles, degrees=False).as_quat()
    permuted_XZX = np.vstack([-q3,-q1,-q2,q0,]).T
    np.isclose(q_XZX,permuted_XZX,atol=atol)

    intrinsic_xzx = 'YXY'
    q_YXY = Rotation.from_euler(intrinsic_xzx, euler_angles, degrees=False).as_quat()
    permuted_YXY = np.vstack([-q2,-q3,-q1,q0,]).T
    np.isclose(q_YXY,permuted_YXY,atol=atol)


def test_quaternion_to_S2():
    n_euler_anlges = 3
    n_ransom_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    s2_mem = quaternion_to_S2(raw_qs)

    s2 = convert_euler_to_S2(euler_angles)

    assert np.allclose(s2, s2_mem, atol=atol)


def test_collapse_to_half_space():
    ''' Tests the collapse_to_half_space function. 
    
    Can convert from EUler angles directly to S2, instead of going through quaternions.'''
    n_euler_anlges = 3
    n_ransom_samples = 7
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))
    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    q_mem, is_mirrored_mem = collapse_to_half_space(raw_qs)
    s2_mem_collapsed = quaternion_to_S2(q_mem)

    s2, is_mirrored = collapse_to_half_space_euler_angles(euler_angles)

    assert np.allclose(s2, s2_mem_collapsed, atol=atol)
    assert np.allclose(is_mirrored_mem, is_mirrored)


def test_q2Spider():
    n_euler_anlges = 3
    n_ransom_samples = 1000

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    phi, theta, psi, = euler_angles.T
    raw_qs = eul_to_quat(phi, theta, psi, flip=True)

    spider_mem = np.array([q2Spider(raw_q) for raw_q in raw_qs.T])
    spider = qs_to_spider(raw_qs)

    random_sample_pass_ratio = 0.4 # around 0.5 in expectation. use lower threshold to account for finite sample size
    assert np.isclose(spider,spider_mem).all(axis=1).mean() > random_sample_pass_ratio


def test_psi_ang():
    n_euler_anlges = 3
    n_ransom_samples = 1000

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    s2_mem = quaternion_to_S2(raw_qs)
    projection_direction_euler_angles_mem = np.array([psi_ang(s2) for s2 in s2_mem.T]).T

    s2 = convert_euler_to_S2(euler_angles)
    projection_direction_euler_angles, projection_direction_euler_angles_alternate = convert_S2_to_euler(s2)

    random_sample_pass_ratio = 0.5 # around 0.5 in expectation. use lower threshold to account for finite sample size
    np.isclose(projection_direction_euler_angles, projection_direction_euler_angles_mem, atol=1e-4).all(axis=1).mean() > random_sample_pass_ratio
    convention_1 = np.isclose(np.mod(np.rad2deg(euler_angles)[:,:2],360), np.mod(projection_direction_euler_angles.T[:,:2],360)).all(1)
    convention_2 = np.isclose(np.mod(np.rad2deg(euler_angles)[:,:2],360), np.mod(projection_direction_euler_angles_alternate.T[:,:2],360)).all(1)
    assert np.logical_xor(convention_1,convention_2).all()
    np.isclose(np.mod(np.rad2deg(euler_angles[:,:2]), 360),projection_direction_euler_angles_mem[:2].T).all(1).mean() > random_sample_pass_ratio







