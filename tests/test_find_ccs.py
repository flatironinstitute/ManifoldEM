import numpy as np
import pytest

from ManifoldEM.optical_flow import get_cluster_nodes_edges


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
