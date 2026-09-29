"""Eigenvectors and singular vectors carry a fixed sign (D48), so results don't depend on which sign a
solver happened to return."""
import numpy as np

from ManifoldEM.core import largest_entry_signs, svdRF
from ManifoldEM.DMembeddingII import diffusion_map_embedding


def _largest_entries(vectors):
    return vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])]


def test_largest_entry_signs():
    vectors = np.array([[1.0, -3.0, 0.5], [-2.0, 1.0, 0.2]])
    np.testing.assert_array_equal(largest_entry_signs(vectors), [-1.0, -1.0, 1.0])
    assert (_largest_entries(vectors * largest_entry_signs(vectors)) > 0).all()


def test_svd_has_fixed_signs_and_the_same_product():
    a = np.random.default_rng(0).normal(size=(30, 6))
    for matrix in (a, a.T):  # both branches of svdRF
        u, s, v = svdRF(matrix)
        assert (_largest_entries(u) > 0).all()
        np.testing.assert_allclose(u @ s @ v.T, matrix, atol=1e-10)


def test_diffusion_map_does_not_depend_on_the_solver_sign(monkeypatch):
    x = np.random.default_rng(1).normal(size=(60, 3))
    distances = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    psi = diffusion_map_embedding(distances, 60, 3)[1]

    # a solver that returns every other eigenvector with the opposite sign (flipping all of them
    # wouldn't change psi, which is a ratio of eigenvectors)
    import ManifoldEM.DMembeddingII as dm
    eigsh = dm.eigsh

    def flipped_eigsh(*args, **kwargs):
        vals, vecs = eigsh(*args, **kwargs)
        return vals, vecs * np.where(np.arange(vecs.shape[1]) % 2, -1.0, 1.0)

    monkeypatch.setattr(dm, "eigsh", flipped_eigsh)
    np.testing.assert_array_equal(diffusion_map_embedding(distances, 60, 3)[1], psi)
