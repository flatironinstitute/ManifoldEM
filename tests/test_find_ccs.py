import h5py
import numpy as np
import pytest

from ManifoldEM.data_store import Anchor, Sense
from ManifoldEM.optical_flow import find_conformational_coords, get_cluster_nodes_edges, write_conformational_coords


def _components(*node_lists):
    """Connected components as FindCCGraph/prune build them: nodes and the numbers of their edges."""
    return [dict(originalNodes=np.array(nodes), originalEdgeList=[np.arange(len(nodes) - 1)]) for nodes in node_lists]


def test_trashed_prds_need_no_anchor():
    # pruning leaves PrD 3 (trashed) as its own component
    components = _components([0, 1, 2], [3])
    nodes, _, no_anchor = get_cluster_nodes_edges(components, [0], 2, allow_empty_clusters=False, trash_ids={3})
    assert list(nodes) == [0, 1, 2]
    assert no_anchor == [1]  # BP leaves it without a coordinate


def test_a_component_without_an_anchor_is_an_error():
    components = _components([0, 1, 2], [3, 4])
    with pytest.raises(ValueError, match=r"\[3, 4\]"):
        get_cluster_nodes_edges(components, [0], 2, allow_empty_clusters=False, trash_ids={3})
    # unless explicitly allowed
    _, _, no_anchor = get_cluster_nodes_edges(components, [0], 2, allow_empty_clusters=True)
    assert no_anchor == [1]


def _analysis_file(path, n_prds):
    with h5py.File(path, "w") as f:
        for i in range(n_prds):
            f.create_group(f"prd_{i}")
    return str(path)


def test_rerun_replaces_the_flow_data(tmp_path):
    out = _analysis_file(tmp_path / "prd_analysis.h5", 2)
    for value in (1.0, 2.0):
        write_conformational_coords(out, np.array([3, -1]), np.array([1, 0]), {0: dict(flow_map=dict(Vx=np.full(4, value)))})

    with h5py.File(out, "r") as f:
        assert sorted(f["prd_0"]) == ["flow_data", "psinum", "sense"]
        assert (f["prd_0/flow_data/flow_map/Vx"][:] == 2.0).all()
        assert (f["prd_1/psinum"][()], f["prd_1/sense"][()]) == (-1, 0)

    # PrD 0 trashed in a later run: its flow data goes with the run it came from
    write_conformational_coords(out, np.array([-1, -1]), np.array([0, 0]), {})
    with h5py.File(out, "r") as f:
        assert "flow_data" not in f["prd_0"]


class _AllAnchored:
    """PrDs 0 and 1 anchored, PrD 2 trashed: nothing left to propagate."""
    anchors = {0: Anchor(2, Sense.REV), 1: Anchor(1, Sense.FWD)}
    anchor_ids = [0, 1]
    trash_ids = {2}

    def prune_graphs(self):
        return dict(Nodes=np.arange(3), nNodes=3, NodesConnComp=[[0, 1], [2]]), []


def test_all_anchored_prds_are_written(tmp_path):
    out = _analysis_file(tmp_path / "prd_analysis.h5", 3)
    find_conformational_coords(_AllAnchored(), None, None, None, 8, output_file=out)

    with h5py.File(out, "r") as f:
        assert [f[f"prd_{i}/psinum"][()] for i in range(3)] == [1, 0, -1]
        assert [f[f"prd_{i}/sense"][()] for i in range(3)] == [-1, 1, 0]
