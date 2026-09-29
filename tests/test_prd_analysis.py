import os

import mrcfile
import numpy as np
import pytest

from ManifoldEM.data_store import data_store
from ManifoldEM.interfaces.simple import decompose
from ManifoldEM.params import params
from ManifoldEM.prd_analysis import (
    get_transformed_images_CTFs_and_image_data,
    psi_analysis_single,
    run_nlsa_second_pass,
)


def test_second_pass_reproduces_the_first_on_a_trimmed_prd(make_project):
    """The trajectory's second NLSA pass recomputes the first pass's movie for the chosen psi, including
    for a PrD whose manifold trimming dropped images (posPath shorter than the PrD)."""
    params.load(str(make_project("t")))
    data_store.get_prds(reload=True)  # bin this project, not whichever an earlier test left cached
    params.ncpu = 1
    decompose(prd_list=[0])

    # drop some images, as the trimming would, keeping psi's rows aligned with posPath
    with data_store.open_analysis("a") as f:
        embedding = f["prd_0/embedding_data"]
        keep = np.arange(7, len(embedding["posPath"]))
        pos_path, psi = embedding["posPath"][:][keep], embedding["psi"][:][keep]
        for name, value in (("posPath", pos_path), ("psi", psi)):
            del embedding[name]
            embedding[name] = value
        f["prd_0/psinum"] = 1
        distances, mask = f["prd_0/distances/D"][:], f["prd_0/image_data/mask"][:]

    images, ctfs, _ = get_transformed_images_CTFs_and_image_data(0)
    (first,) = psi_analysis_single(distances, images, ctfs, mask, psi, [1], pos_path,
                                   params.con_order_range, params.num_psi_truncated)
    second = run_nlsa_second_pass(0, params.analysis_file, dtype=np.float64)

    assert second.shape[0] == len(first["tau"])
    # IMG1 is stored as float16 (rounding error up to ~1e-3 here); without the posPath fix the frames differ by ~4
    np.testing.assert_allclose(second[first["tauinds"]], first["IMG1"].astype(float), atol=5e-3)


def test_parallel_decompose_of_selected_prds_writes_each_to_its_own_group(make_project):
    params.load(str(make_project("p")))
    prds = data_store.get_prds(reload=True)
    params.ncpu = 2
    decompose(prd_list=[3, 1])

    with data_store.open_analysis() as f:
        assert sorted(f) == ["prd_1", "prd_3"]
        for i in (1, 3):
            assert len(f[f"prd_{i}/image_data/rotations"]) == len(prds.thresholded_image_indices[i])


def test_decompose_reads_float16_stacks(make_project):
    params.load(str(make_project("h")))
    data_store.get_prds(reload=True)
    with mrcfile.open(params.img_stack_file, "r+") as m:
        m.set_data(m.data.astype(np.float16))
    params.ncpu = 1
    decompose(prd_list=[0])

    with data_store.open_analysis() as f:
        assert np.isfinite(f["prd_0/distances/D"][:]).all()


def test_prd_data_reads_the_decomposition(make_project):
    params.load(str(make_project("v")))
    data_store.get_prds(reload=True)
    params.ncpu = 1
    decompose(prd_list=[0])

    prd = data_store.get_prd_data(0)
    n_images = len(prd.info.raw_image_indices)
    assert prd.dist_data["D"].shape == (n_images, n_images)
    assert prd.psi_data["psi"].shape[0] == len(prd.psi_data["posPath"])
    images, _, _ = get_transformed_images_CTFs_and_image_data(0)
    np.testing.assert_array_equal(prd.transformed_images, images)  # the pipeline's own transform
    with pytest.raises(ValueError, match="find-ccs"):
        prd.EL_data


def test_steps_after_decompose_say_what_to_run(make_project):
    params.load(str(make_project("o")))
    data_store.get_prds(reload=True)
    with pytest.raises(FileNotFoundError, match="decompose"):
        data_store.require_analysis()

    # a project from before prd_analysis.h5
    os.makedirs(os.path.join(params.out_dir, "distances"))
    with pytest.raises(FileNotFoundError, match="earlier version of ManifoldEM"):
        data_store.get_prd_data(0)
