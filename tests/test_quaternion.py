import numpy as np

from ManifoldEM.util import eul_to_quat
from scipy.spatial.transform import Rotation

def test_manifold_em_raw_quaternion():
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




