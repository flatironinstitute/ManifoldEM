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
    np.random.seed(0)

    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_ransom_samples, n_euler_anlges))

    intrinsic_zxz = 'ZXZ'
    intrinsic_zyz = 'ZYZ'
    q_ZXZ = Rotation.from_euler(intrinsic_zxz, euler_angles, degrees=False).as_quat()
    q_ZYZ = Rotation.from_euler(intrinsic_zyz, euler_angles, degrees=False).as_quat()

    phi, theta, psi, = euler_angles.T
    flip = True
    raw_qs = eul_to_quat(phi, theta, psi, flip=flip)
    q0, q1, q2, q3 = raw_qs

    permuted_ZXZ = np.vstack([-q2, q1, -q3, q0,]).T
    assert np.allclose(q_ZXZ,permuted_ZXZ)

    permuted_ZYZ = np.vstack([-q1, -q2, -q3, q0,]).T
    assert np.isclose(q_ZYZ,permuted_ZYZ)




