import sys

import toml

from ManifoldEM.interfaces import cli
from ManifoldEM.params import params


def _manifold_cli(monkeypatch, *argv):
    """Run `manifold-cli <argv>` in this process, starting from the default params like a new process."""
    vars(params).clear()
    monkeypatch.setattr(sys, "argv", ["manifold-cli", *argv])
    cli.main()


def _saved(path):
    return toml.load(path)["params"]


def test_bare_threshold_changes_nothing(make_project, monkeypatch):
    path = make_project("c", thres_low=70)
    params.tess_hemisphere_vec = [0.0, 0.0, 1.0]
    params.save(str(path))

    _manifold_cli(monkeypatch, "threshold", str(path))

    saved = _saved(path)
    assert (saved["prd_thres_low"], saved["prd_thres_high"]) == (70, 150)
    assert saved["tess_hemisphere_vec"] == [0.0, 0.0, 1.0]


def test_given_flags_are_saved_before_the_step_runs(make_project, monkeypatch):
    path = make_project("d")
    seen = {}
    # stand-in for the step, which would reload the project file
    monkeypatch.setitem(cli._funcs, "decompose", lambda args: seen.update(_saved(path)))

    _manifold_cli(monkeypatch, "threshold", "--prd_thres_low", "80", "--tess_hemisphere_vec", "0,1,0", str(path))
    _manifold_cli(monkeypatch, "decompose", "--num_psi", "5", str(path))

    assert (seen["prd_thres_low"], seen["tess_hemisphere_vec"], seen["num_psi"]) == (80, [0.0, 1.0, 0.0], 5)
    assert seen["distance_filter_type"] == "Butter"  # not given: the project's value stays
