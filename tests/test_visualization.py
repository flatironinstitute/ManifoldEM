"""ManifoldEM.visualization on a project that has only been binned (`make_project`, see conftest.py).

The viewers that need later pipeline outputs (diffusion maps, CC voting, optical flow, trajectory)
and the interactive widgets aren't covered here: CI installs neither the pipeline outputs nor the
`notebook` extra.
"""
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest
import toml

from ManifoldEM.data_store import Sense
from ManifoldEM.params import ProjectLevel
from ManifoldEM.visualization import load_session


@pytest.fixture
def figures(monkeypatch):
    """The figures the views draw, closed instead of shown."""
    drawn = []

    def show(*args, **kwargs):
        drawn.append(plt.gcf())
        plt.close("all")

    monkeypatch.setattr(plt, "show", show)
    return drawn


def saved(toml_path):
    """The project as saved on disk: its params and its pd_data.pkl."""
    name = toml_path.stem[len("params_"):]
    with open(toml_path.parent / "output" / name / "pd_data.pkl", "rb") as f:
        return toml.load(toml_path)["params"], pickle.load(f)


def test_import_needs_no_notebook_extra():
    code = ("import sys, ManifoldEM.visualization; "
            "print(sorted(m for m in ('plotly', 'ipywidgets', 'IPython') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_load_session_bins_a_new_project(make_project):
    toml_path = make_project("a")
    load_session(toml_path)

    _, pd_data = saved(toml_path)
    assert len(pd_data["thres_ids"]) == 9


def test_static_views(make_project, figures, monkeypatch, capsys):
    toml_path = make_project("a")
    viz = load_session(toml_path)
    monkeypatch.setitem(sys.modules, "plotly", None)  # as if plotly isn't installed
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    viz.show_s2_sphere(interactive=False)
    assert "falling back to matplotlib" in capsys.readouterr().out
    viz.show_thresholds()
    viz.show_thresholds(low=70, high=100)
    assert "preview only" in capsys.readouterr().out
    viz.show_anchors()
    assert "No anchors yet" in capsys.readouterr().out
    assert len(figures) == 4

    params, pd_data = saved(toml_path)
    assert (params["prd_thres_low"], params["prd_thres_high"]) == (60, 150)
    assert (pd_data["thres_low"], pd_data["thres_high"]) == (60, 150)


def test_interactive_views_name_the_notebook_extra(make_project, monkeypatch):
    viz = load_session(make_project("a"))
    monkeypatch.setitem(sys.modules, "ipywidgets", None)  # as if ipywidgets isn't installed

    with pytest.raises(ImportError, match="notebook extra"):
        viz.show_prd_viewer(interactive=True)


def test_anchor_and_trash_edits_are_saved(make_project, figures, capsys):
    toml_path = make_project("a")
    viz = load_session(toml_path)

    viz.set_anchor(prd=2, psi=3, sense="rev")
    viz.set_anchor(prd=4, psi=1)
    _, pd_data = saved(toml_path)
    assert sorted(pd_data["anchors"]) == [1, 3]  # PrD numbers are 1-based
    assert (pd_data["anchors"][1].CC, pd_data["anchors"][1].sense) == (3, Sense.REV)

    viz.remove_anchor(4)
    viz.trash_prd(2, 5)  # trashing PrD 2 also removes its anchor
    _, pd_data = saved(toml_path)
    assert pd_data["anchors"] == {}
    assert pd_data["trash_ids"] == {1, 4}
    with pytest.raises(ValueError, match="in the trash"):
        viz.set_anchor(prd=5, psi=1)

    viz.untrash_prd(5)
    _, pd_data = saved(toml_path)
    assert pd_data["trash_ids"] == {1}

    viz.show_anchors()
    out = capsys.readouterr().out
    assert "trash: PrD 2" in out and "NO ANCHOR" in out

    num_psi = saved(toml_path)[0]["num_psi"]
    for bad in (dict(prd=0, psi=1), dict(prd=10, psi=1), dict(prd=1, psi=num_psi + 1), dict(prd=1, psi=1, sense="UP")):
        with pytest.raises(ValueError):
            viz.set_anchor(**bad)


def test_set_thresholds_rebins_and_clears_the_edits(make_project, capsys):
    toml_path = make_project("a")
    viz = load_session(toml_path)
    viz.set_anchor(prd=1, psi=1)
    viz.trash_prd(2)

    viz.set_thresholds(60, 150)
    assert "nothing changed" in capsys.readouterr().out
    with pytest.raises(ValueError, match="at least 3 PrDs"):
        viz.set_thresholds(1000, 2000)

    viz.set_thresholds(70, 150)
    params, pd_data = saved(toml_path)
    assert (params["prd_thres_low"], params["project_level"]) == (70, ProjectLevel.BINNING.value)
    assert len(pd_data["thres_ids"]) == 8
    assert pd_data["anchors"] == {} and pd_data["trash_ids"] == set()


def test_set_path_width(make_project):
    toml_path = make_project("a")
    viz = load_session(toml_path)

    viz.set_path_width(3)
    params, _ = saved(toml_path)
    assert params["width_1D"] == 3
    for bad in (0, params["states_per_coord"] + 1):
        with pytest.raises(ValueError):
            viz.set_path_width(bad)


def test_views_read_the_project_again(make_project, figures, capsys):
    toml_path = make_project("a")
    viz = load_session(toml_path)
    viz.show_anchors()
    capsys.readouterr()

    load_session(toml_path).set_anchor(prd=2, psi=3, sense="REV")  # e.g. from another kernel
    viz.show_anchors()
    assert "PrD 2 ψ3 REV" in capsys.readouterr().out


def test_two_sessions_keep_their_own_edits(make_project):
    a, b = make_project("a"), make_project("b")
    viz_a, viz_b = load_session(a), load_session(b)

    viz_b.set_anchor(prd=1, psi=1)
    viz_b.trash_prd(2)
    viz_a.set_thresholds(70, 150)  # a now has thresholds b doesn't
    viz_b.set_anchor(prd=3, psi=1)

    _, pd_b = saved(b)
    assert sorted(pd_b["anchors"]) == [0, 2]
    assert pd_b["trash_ids"] == {1}
    _, pd_a = saved(a)
    assert len(pd_a["thres_ids"]) == 8
