import numpy as np
from scipy.spatial.transform import Rotation

from ManifoldEM.util import eul_to_quat
from ManifoldEM.quaternion import quaternion_to_S2, collapse_to_half_space

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



def convert_euler_to_S2(euler_angles):
    rotation = Rotation.from_euler('ZXZ', euler_angles, degrees=False).as_matrix()
    rxz, ryz, rzz = rotation[:,:,-1].T
    s2 = np.stack([-ryz,rxz,rzz], axis=1).T
    return s2

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


def collapse_to_half_space_euler_angles(euler_angles):
    s2 = convert_euler_to_S2(euler_angles)
    is_mirrored = s2[0,:] < 0.0
    s2[:,is_mirrored] = -s2[:,is_mirrored]
    return s2, is_mirrored


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
    s2_mem = quaternion_to_S2(raw_qs).T
    q_mem, is_mirrored_mem = collapse_to_half_space(raw_qs)
    s2_mem_collapsed = quaternion_to_S2(q_mem)

    s2, is_mirrored = collapse_to_half_space_euler_angles(euler_angles)

    assert np.allclose(s2, s2_mem_collapsed, atol=atol)
    assert np.allclose(is_mirrored_mem, is_mirrored)








