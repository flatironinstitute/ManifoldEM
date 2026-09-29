"""The whole pipeline, decompose to trajectory, on the tiny test project (8-px noise images), and the
notebook's viewers on its output."""
import base64
import glob
import io
import os

import h5py
import numpy as np
import pytest
from PIL import Image

from conftest import _write_inputs
from ManifoldEM.data_store import data_store
from ManifoldEM.interfaces import simple
from ManifoldEM.params import params


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project that has been through every step, with two processes (the workers read
    prd_analysis.h5 themselves). Returns its params file."""
    folder = tmp_path_factory.mktemp("pipeline")
    _write_inputs(folder, np.random.default_rng(0))
    start = os.getcwd()
    try:
        vars(params).clear()
        (folder / "p").mkdir()
        os.chdir(folder / "p")
        simple.init("p", "", str(folder / "particles.star"), str(folder / "stack.mrcs"), "",
                    pixel_size=2.0, diameter=100.0, resolution=10.0, aperture_index=1, overwrite=False)
        params.prd_thres_low, params.prd_thres_high, params.ncpu = 60, 150, 2
        params.save()
        data_store.get_prds(reload=True)
        simple.decompose()
        simple.find_conformational_coordinates()
        simple.probability_landscape()
        simple.compute_trajectory()
        yield folder / "p" / "params_p.toml"
    finally:
        os.chdir(start)


def _load(project):
    params.load(str(project))
    return data_store.get_prds(reload=True)


def test_every_step_writes_its_output(project):
    prds = _load(project)
    with data_store.open_analysis() as f:
        assert sorted(f) == [f"prd_{i}" for i in range(prds.n_thresholded)]
        psinums = [int(f[f"prd_{i}/psinum"][()]) for i in range(prds.n_thresholded)]
        assert all(0 <= p < params.num_psi for p in psinums)
        assert all("flow_data" in f[f"prd_{i}"] for i in range(prds.n_thresholded))

    with np.load(params.probability_landscape_file) as landscape:
        assert len(landscape["occupancy"]) == params.states_per_coord
        assert landscape["occupancy"].sum() == landscape["tau_avg"].size

    stars = glob.glob(os.path.join(params.bin_dir, f"EulerAngles_{params.traj_name}_*_of_{params.states_per_coord}.star"))
    stacks = glob.glob(os.path.join(params.bin_dir, f"imgsRELION_{params.traj_name}_*_of_{params.states_per_coord}.mrcs"))
    assert len(stars) == len(stacks) > 0
    angles = np.array([[float(v) for v in line.split()[1:4]] for s in stars for line in open(s) if "@" in line])
    assert np.isfinite(angles).all() and len(angles) > 0


def test_notebook_viewers_render_the_output(project, monkeypatch):
    import matplotlib.pyplot as plt
    from ManifoldEM.visualization import load_session

    shown = []  # the number of figures open at each plt.show()

    def show(*args, **kwargs):
        shown.append(len(plt.get_fignums()))
        plt.close("all")

    monkeypatch.setattr(plt, "show", show)
    viz = load_session(project)
    viz.show_prd_viewer(default_prd=2, interactive=False)
    viz.show_diffusion_map(default_prd=2, interactive=False)
    viz.show_cc_voting(interactive=False)
    viz.show_optical_flow(default_prd=2, default_psi=1, interactive=False)
    viz.show_trajectory()
    assert len(shown) == 7 and all(shown)

    # no viewer keeps prd_analysis.h5 open, which would lock the CLI out of it
    with h5py.File(params.analysis_file, "a"):
        pass


def test_psi_movies_are_animated_gifs(project):
    from ManifoldEM.visualization import VizSession

    _load(project)
    movies = VizSession._psi_movie_frames(1)
    assert len(movies) == params.num_psi
    html = VizSession._gif_html(movies[0], "psi_1")
    gif = Image.open(io.BytesIO(base64.b64decode(html.split("base64,")[1].split('"')[0])))
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info["duration"])
    # 5 frames per second; Pillow merges repeated frames and adds up their durations
    assert gif.n_frames > 1 and sum(durations) == 200 * len(movies[0])
