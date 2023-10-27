import os

from enum import Enum
import numpy as np
import pickle

from typing import List, Any, Tuple, Dict, Set
from nptyping import NDArray, Shape, Int, Int64, Float64

from ManifoldEM.params import p
from ManifoldEM.Data import get_from_relion
from ManifoldEM.util import augment
from ManifoldEM.S2tessellation import bin_and_threshold
from ManifoldEM.FindCCGraph import op as FindCCGraph


class Sense(Enum):
    FWD = 1
    REV = -1

    @staticmethod
    def from_index(idx: int) -> 'Sense':
        if idx == 0:
            return Sense.FWD
        if idx == 1:
            return Sense.REV
        raise ValueError("Invalid index")

    def to_index(self) -> int:
        return 0 if self == Sense.FWD else 1


class Anchor:
    def __init__(self, CC: int = 1, sense: Sense = Sense.FWD):
        self.CC: int = CC
        self.sense: Sense = sense


class _ProjectionDirections:
    def __init__(self):
        self.thres_low: int = p.PDsizeThL
        self.thres_high: int = p.PDsizeThH
        self.bin_centers: NDArray[Shape["3,*", Any], Float64] = np.empty(shape=(3, 0))

        self.defocus: NDArray[Shape["*"], Float64] = np.empty(0)
        self.microscope_origin: Tuple[NDArray[Shape["*"], Float64],
                                      NDArray[Shape["*"], Float64]] = (np.empty(0), np.empty(0))

        self.pos_full: NDArray[Shape["3", Any], Float64] = np.empty(shape=(3,0))
        self.quats_full: NDArray[Shape["4", Any], Float64] = np.empty(shape=(4,0))

        self.image_indices_full: NDArray[Shape["*"], List[Int]] = np.empty(0, dtype=object)
        self.thres_ids: NDArray[Shape["*"], Int64] = np.empty(0, dtype=np.int64)
        self.occupancy_full: NDArray[Shape["*"], Int] = np.empty(0, dtype=int)

        self.anchors: Dict[int, Anchor] = {}
        self.trash_ids: Set[int] = set()
        self.reembed_ids: Set[int] = set()

        self.neighbor_graph: Dict[str, Any] = {}
        self.neighbor_subgraph: List[Dict[str, Any]] = []

        self.neighbor_graph_pruned: Dict[str, Any] = {}
        self.neighbor_subgraph_pruned: List[Dict[str, Any]] = []

        self.pos_thresholded: NDArray[Shape["3", Any], Float64] = np.empty(shape=(3,0))
        self.theta_thresholded: NDArray[Shape["*"], Float64] = np.empty(0)
        self.phi_thresholded: NDArray[Shape["*"], Float64] = np.empty(0)
        self.cluster_ids: NDArray[Shape["*"], Int] = np.empty(0, dtype=int)


    def load(self, pd_file=None):
        if pd_file is None:
            pd_file = p.pd_file

        with open(pd_file, 'rb') as f:
            self.__dict__.update(pickle.load(f))

    
    def save(self):
        with open(p.pd_file, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)


    def update(self):
        # Load if cache exists and store uninitialized
        if self.pos_full.size == 0 and os.path.isfile(p.pd_file):
            self.load(p.pd_file)

        # If uninitialized or things have changed, actually update
        force_rebuild = bool(os.environ.get('MANIFOLD_REBUILD_DS', 0))
        if force_rebuild or self.pos_full.size == 0 or self.thres_low != p.PDsizeThL or self.thres_high != p.PDsizeThH:
            if force_rebuild:
                print("Rebuilding data store")
                os.environ.pop('MANIFOLD_REBUILD_DS')

            print("Calculating projection direction information")
            sh, q, U, V = get_from_relion(p.align_param_file, flip=True)
            df = (U + V) / 2

            # double the number of data points by augmentation
            q = augment(q)
            df = np.concatenate((df, df))

            image_indices, pos_full, bin_centers, occupancy, conjugate_bin_ids = \
                bin_and_threshold(q, p.ang_width, p.PDsizeThL, p.PDsizeThH)

            self.thres_low = p.PDsizeThL
            self.thres_high = p.PDsizeThH

            self.bin_centers = bin_centers
            self.defocus = df
            self.microscope_origin = sh

            self.pos_full = pos_full
            self.quats_full = q

            self.image_indices_full = image_indices
            self.thres_ids = conjugate_bin_ids
            self.occupancy_full = occupancy

            self.anchors = {}
            self.trash_ids = set()

            self.pos_thresholded = self.bin_centers[:, self.thres_ids]
            self.phi_thresholded = np.arctan2(self.pos_thresholded[1, :], self.pos_thresholded[0, :]) * 180. / np.pi
            self.theta_thresholded = np.arccos(self.pos_thresholded[2, :]) * 180. / np.pi

            self.neighbor_graph, self.neighbor_subgraph = \
                FindCCGraph(self.thresholded_image_indices, self.n_bins, self.pos_thresholded)

            def get_cluster_ids(G):
                nodesColor = np.zeros(G['nNodes'], dtype='int')
                for i, nodesCC in enumerate(G['NodesConnComp']):
                    nodesColor[nodesCC] = i

                return nodesColor

            self.cluster_ids = get_cluster_ids(self.neighbor_graph)

            p.numberofJobs = len(self.thres_ids)

            p.save()
            self.save()


    def insert_anchor(self, id: int, anchor: Anchor):
        self.anchors[id] = anchor


    def remove_anchor(self, id: int):
        if id in self.anchors:
            self.anchors.pop(id)


    def deduplicate(self, arr):
        mid = arr.shape[-1] // 2
        if 2 * mid == arr.shape[-1]:
            return arr[:mid]
        else:
            return arr[mid:]

    @property
    def occupancy_no_duplication(self):
        return self.deduplicate(self.occupancy_full)


    @property
    def bin_centers_no_duplication(self):
        mid = self.bin_centers.shape[1] // 2
        if 2 * mid == self.bin_centers.shape[1]:
            return self.bin_centers[:, :mid]
        else:
            return self.bin_centers[:, mid:]


    @property
    def anchor_ids(self):
        return sorted(list(self.anchors.keys()))


    @property
    def thresholded_image_indices(self):
        return self.image_indices_full[self.thres_ids]


    @property
    def occupancy(self):
        return self.occupancy_full[self.thres_ids]


    @property
    def n_bins(self):
        return self.bin_centers.shape[1]


    @property
    def n_thresholded(self):
        return len(self.thres_ids)


class _DataStore:
    _projection_directions = _ProjectionDirections()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(_DataStore, cls).__new__(cls)
        return cls.instance


    def get_prds(self):
        self._projection_directions.update()
        return self._projection_directions


data_store = _DataStore()
