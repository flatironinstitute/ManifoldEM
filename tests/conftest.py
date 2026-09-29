"""A tiny ManifoldEM project for tests: synthetic particle orientations, set up by `manifold-cli init`."""
import os

import matplotlib
import mrcfile
import numpy as np
import pytest

matplotlib.use("Agg")  # the notebook views draw figures; never open windows

from ManifoldEM.params import params


def _write_inputs(folder, rng):
    """Write `particles.star` and a matching stack of blank 8-px images.

    Binning reads only the orientations: 8 clusters of 150 particles along an arc on S2 (tilt 60°,
    rot 0°–63°). With the settings below, that makes 9 PrDs over a low threshold of 60 (8 over 70),
    in one connected neighbour graph.
    """
    cols = ["rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast", "rlnDefocusU", "rlnDefocusV",
            "rlnOriginX", "rlnOriginY", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    rows = []
    for rot in range(0, 72, 9):
        for _ in range(150):
            defocus = rng.uniform(15000, 25000)
            rows.append(f"300.0 2.7 0.1 {defocus:.1f} {defocus:.1f} 0.0 0.0 "
                        f"{rot + rng.normal(0, 1):.4f} {60 + rng.normal(0, 1):.4f} {rng.uniform(0, 360):.4f}")
    header = ["", "data_images", "", "loop_"] + [f"_{c} #{i}" for i, c in enumerate(cols, 1)]
    (folder / "particles.star").write_text("\n".join(header + rows) + "\n")
    with mrcfile.new(str(folder / "stack.mrcs")) as m:
        m.set_data(np.zeros((len(rows), 8, 8), dtype=np.float32))
        m.set_image_stack()


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """Return a function that sets up a project like `manifold-cli init` and returns its params file.

    The particles aren't binned yet; the first `data_store.get_prds()` does that. Each project starts
    from the default `params`. Loading a project changes the working directory, so the fixture
    restores it afterwards.
    """
    from ManifoldEM.interfaces.interactive import init

    _write_inputs(tmp_path, np.random.default_rng(0))
    monkeypatch.chdir(tmp_path)

    def make(name, thres_low=60, thres_high=150):
        vars(params).clear()  # back to the class defaults
        project_dir = tmp_path / name
        project_dir.mkdir()
        os.chdir(project_dir)
        init(name, "", str(tmp_path / "particles.star"), str(tmp_path / "stack.mrcs"), "",
             pixel_size=2.0, diameter=100.0, resolution=10.0, aperture_index=1, overwrite=False)
        params.prd_thres_low, params.prd_thres_high = thres_low, thres_high
        params.save()
        return project_dir / f"params_{name}.toml"

    return make
