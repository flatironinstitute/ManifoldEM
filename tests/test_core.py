import numpy as np

from ManifoldEM.core import get_euler_from_PD, project_mask, euler_rot_matrix_3D_spider
from ManifoldEM.quaternion import convert_euler_to_S2, convert_S2_to_euler
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform


def test_euler_rot_matrix():
    n_euler_angles = 3
    n_random_samples = 100
    atol = 1e-4

    np.random.seed(0)
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))

    rotmat_scipy = Rotation.from_euler('zyz', -euler_angles).as_matrix()
    rotmat_mem = np.array([euler_rot_matrix_3D_spider(e[0], e[1], e[2]) for e in euler_angles])

    assert np.allclose(rotmat_scipy, rotmat_mem, atol=atol)


def test_get_euler_from_PD():
    np.random.seed(0)
    n_random_samples = 1000
    n_euler_angles = 3
    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, n_euler_angles))
    s2 = convert_euler_to_S2(euler_angles)

    euler_back = np.array([get_euler_from_PD(r) for r in s2.T])
    s2_back = convert_euler_to_S2(euler_back)

    assert np.allclose(s2, s2_back)


def test_project_mask():
    vol = np.zeros((40, 40, 40))
    vol[5:10, 25:30, 20:25] = 1.0
    n_random_samples = 100
    np.random.seed(0)

    euler_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n_random_samples, 3))
    s2 = convert_euler_to_S2(euler_angles).T
    euler_back = np.array([get_euler_from_PD(r) for r in s2])

    dims = vol.shape
    c_in = 0.5 * np.array(dims)
    c_out = 0.5 * np.array(dims)

    euler_new = convert_S2_to_euler(s2.T).T

    def project_mask_test(vol, angs):
        angs = np.flip(angs)

        rotmat = Rotation.from_euler('zyz', angs, degrees=False).as_matrix()
        cen_offset = c_in - np.dot(rotmat, c_out)
        rho = affine_transform(input=np.swapaxes(vol, 0, 2),
                               matrix=rotmat,
                               offset=cen_offset,
                               output_shape=dims,
                               mode='nearest')

        return np.sum(rho, axis=2).reshape(40, 40).T > 1

    for i in range(n_random_samples):
        mask_old = project_mask(vol, s2[i])
        # Make sure projection mask test is the same as production version
        mask_new = project_mask_test(vol, euler_back[i])

        assert np.allclose(mask_old, mask_new)

        # Check that our S2 conversion gives the same result
        mask_new = project_mask_test(vol, euler_new[i])
        assert np.allclose(mask_old, mask_new)

