import numpy as np

from ManifoldEM.belief_propagation import initialize_anchor_node_measures
from ManifoldEM.data_store import Anchor, Sense


def test_anchor_state_index():
    """
    Anchor.CC counts psi from 1 (psi_1.gif, the Anchor default), while belief
    propagation uses 0-based states with reverse senses offset by the number of psis.
    """
    num_psis = 8
    # an anchor on psi_1 is psinum 0
    assert Anchor(1, Sense.FWD).state_index(num_psis) == 0
    assert Anchor(1, Sense.REV).state_index(num_psis) == num_psis
    # the last psi stays inside the 2 * num_psis state vector
    assert Anchor(num_psis, Sense.FWD).state_index(num_psis) == num_psis - 1
    assert Anchor(num_psis, Sense.REV).state_index(num_psis) == 2 * num_psis - 1


def test_anchor_node_measures_use_the_anchor_state():
    """Belief propagation puts each anchor's potential on the state of its psi and sense."""
    num_psis = 8
    anchors = {3: Anchor(1, Sense.FWD), 5: Anchor(num_psis, Sense.FWD), 7: Anchor(num_psis, Sense.REV)}
    nodes, measures = initialize_anchor_node_measures(anchors, num_psis, 110.0)

    assert nodes == [3, 5, 7]  # forward-sensed anchors first
    assert measures.shape == (2 * num_psis, 3)
    for column, node in enumerate(nodes):
        assert list(np.nonzero(measures[:, column])[0]) == [anchors[node].state_index(num_psis)]
