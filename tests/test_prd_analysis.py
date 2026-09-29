import numpy as np

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
