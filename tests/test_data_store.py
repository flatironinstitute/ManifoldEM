"""data_store: `get_prds(reload=True)` when more than one project is loaded in a process, and the guessed anchors."""
import os
import pickle

import numpy as np

from ManifoldEM.data_store import Anchor, Sense, data_store
from ManifoldEM.params import params


def test_reload_keeps_the_edits_of_each_project(make_project):
    a, b = make_project("a"), make_project("b")

    params.load(str(b))
    prds = data_store.get_prds(reload=True)  # bins b
    prds.insert_anchor(0, Anchor(1, Sense.FWD))
    prds.trash_ids.add(1)
    prds.save()

    params.load(str(a))
    params.prd_thres_low = 70
    data_store.get_prds(reload=True)  # bins a, whose thresholds now differ from b's

    # Without reload, the cache still holds a, so b would be re-binned, which clears its anchors and trash
    params.load(str(b))
    prds = data_store.get_prds(reload=True)
    assert list(prds.anchors) == [0]
    assert prds.trash_ids == {1}


def test_reload_bins_a_project_that_has_no_pd_file_yet(make_project):
    a, b = make_project("a"), make_project("b")  # same thresholds

    params.load(str(a))
    data_store.get_prds(reload=True)
    params.load(str(b))
    data_store.get_prds(reload=True)

    # without reload, b would reuse a's cached bins, since the thresholds match, and never write its own file
    assert os.path.isfile(params.pd_file)


def test_reload_picks_up_changes_saved_elsewhere(make_project):
    params.load(str(make_project("a")))
    data_store.get_prds(reload=True)

    # another process, e.g. the notebook in another kernel, anchors PrD 3
    with open(params.pd_file, "rb") as f:
        saved = pickle.load(f)
    saved["anchors"] = {2: Anchor(4, Sense.REV)}
    with open(params.pd_file, "wb") as f:
        pickle.dump(saved, f)

    anchors = data_store.get_prds(reload=True).anchors
    assert list(anchors) == [2]
    assert (anchors[2].CC, anchors[2].sense) == (4, Sense.REV)


def test_guessed_anchors_keep_existing_anchors(make_project):
    params.load(str(make_project("g")))
    prds = data_store.get_prds(reload=True)
    top = int(np.argmax(prds.occupancy))  # one component: the guess goes to the most occupied PrD
    prds.insert_anchor(top, Anchor(3, Sense.REV))

    prds.guess_and_mark_anchors()

    assert list(prds.anchors) == [top]
    assert (prds.anchors[top].CC, prds.anchors[top].sense) == (3, Sense.REV)
