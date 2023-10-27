import numpy as np

from ManifoldEM.S2tessellation import bin_and_threshold, quaternion_to_S2
from ManifoldEM.core import distribute3Sphere
from ManifoldEM.util import eul_to_quat

def cart_to_polar(u):
    theta = np.arccos(u[:,2])
    phi = np.arccos(u[:,0] / np.sqrt(u[:,0]**2 + u[:,1]**2)) * np.sign(u[:,1])
    return theta, phi


def test_S2_binning():
    ang_width = np.pi / 5
    num_bins = int(4 * np.pi / (ang_width**2))
    thres_low = 2
    thres_high = 5

    bin_centers, _ = distribute3Sphere(num_bins)
    theta, phi = cart_to_polar(bin_centers)
    psi = np.zeros_like(theta)
    q = eul_to_quat(phi, theta, psi, False)
    bin_centers_new = quaternion_to_S2(q).T

    assert np.allclose(bin_centers, bin_centers_new), "unit_vec->euler_angle->quat->unit_vec != unit_vec"

    test_points = np.vstack((bin_centers, bin_centers, bin_centers))
    theta, phi = cart_to_polar(test_points)
    psi = np.zeros_like(theta)
    q = eul_to_quat(phi, theta, psi, False)

    neighb_list, S2, bin_centers, n_points_in_bin, _ = bin_and_threshold(q, ang_width, thres_low, thres_high)
    assert np.allclose(S2[:,0:num_bins], bin_centers), "bin_center doesn't match"
    assert np.all(n_points_in_bin == 3), "invalid bin count"
    assert np.all(np.arange(3*num_bins).reshape(3, num_bins).transpose() == neighb_list), "invalid neighbor list"
