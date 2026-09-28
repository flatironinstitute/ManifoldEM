from ManifoldEM.data_store import Anchor, Sense


def test_anchor_state_index():
    """
    Anchor.CC counts psi from 1 (psi_1.gif, the GUI spin box, the Anchor default), while belief
    propagation uses 0-based states with reverse senses offset by the number of psis.
    """
    num_psis = 8
    # an anchor on psi_1 is psinum 0
    assert Anchor(1, Sense.FWD).state_index(num_psis) == 0
    assert Anchor(1, Sense.REV).state_index(num_psis) == num_psis
    # the last psi stays inside the 2 * num_psis state vector
    assert Anchor(num_psis, Sense.FWD).state_index(num_psis) == num_psis - 1
    assert Anchor(num_psis, Sense.REV).state_index(num_psis) == 2 * num_psis - 1
