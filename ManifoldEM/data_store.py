import os

from enum import Enum
import numpy as np
import pickle

from typing import List, Any, Tuple, Dict, Set
from nptyping import NDArray, Shape, Int, Int64, Float64, Bool

from ManifoldEM.params import params
from ManifoldEM.star import get_align_data
from ManifoldEM.quaternion import collapse_to_half_space, quaternion_to_S2
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
        self.thres_low: int = params.prd_thres_low
        self.thres_high: int = params.prd_thres_high
        self.bin_centers: NDArray[Shape["3,*"], Float64] = np.empty(shape=(3, 0))

        self.defocus: NDArray[Shape["*"], Float64] = np.empty(0)
        self.microscope_origin: Tuple[NDArray[Shape["*"], Float64],
                                      NDArray[Shape["*"], Float64]] = (np.empty(0), np.empty(0))

        self.pos_raw: NDArray[Shape["3,*"], Float64] = np.empty(shape=(3,0))
        self.pos_full: NDArray[Shape["3,*"], Float64] = np.empty(shape=(3,0))
        self.quats_raw: NDArray[Shape["4,*"], Float64] = np.empty(shape=(4,0))
        self.quats_full: NDArray[Shape["4,*"], Float64] = np.empty(shape=(4,0))

        self.image_indices_full: NDArray[Shape["*"], Any] = np.empty(0, dtype=object)
        self.thres_ids: list[int] = []
        self.occupancy_full: NDArray[Shape["*"], Int] = np.empty(0, dtype=int)

        self.anchors: Dict[int, Anchor] = {}
        self.trash_ids: Set[int] = set()
        self.reembed_ids: Set[int] = set()

        self.neighbor_graph: Dict[str, Any] = {}
        self.neighbor_subgraph: List[Dict[str, Any]] = []

        self.neighbor_graph_pruned: Dict[str, Any] = {}
        self.neighbor_subgraph_pruned: List[Dict[str, Any]] = []

        self.pos_thresholded: NDArray[Shape["3,*"], Float64] = np.empty(shape=(3,0))
        self.theta_thresholded: NDArray[Shape["*"], Float64] = np.empty(0)
        self.phi_thresholded: NDArray[Shape["*"], Float64] = np.empty(0)
        self.cluster_ids: NDArray[Shape["*"], Int] = np.empty(0, dtype=int)

        self.image_is_mirrored: NDArray[Shape["*"], Bool] = np.empty(0, dtype=bool)


    def load(self, pd_file=None):
        if pd_file is None:
            pd_file = params.pd_file

        with open(pd_file, 'rb') as f:
            self.__dict__.update(pickle.load(f))

    
    def save(self):
        with open(params.pd_file, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)


    def update(self):
        # Load if cache exists and store uninitialized
        if self.pos_full.size == 0 and os.path.isfile(params.pd_file):
            self.load(params.pd_file)

        # If uninitialized or things have changed, actually update
        force_rebuild = bool(os.environ.get('MANIFOLD_REBUILD_DS', 0))
        if force_rebuild or self.pos_full.size == 0 or self.thres_low != params.prd_thres_low or self.thres_high != params.prd_thres_high:
            if force_rebuild:
                print("Rebuilding data store")
                os.environ.pop('MANIFOLD_REBUILD_DS')

            print("Calculating projection direction information")
            self.microscope_origin, self.quats_raw, U, V = get_align_data(params.align_param_file, flip=True)
            df = (U + V) / 2

            plane_vec = np.array(params.tess_hemisphere_vec)
            self.quats_raw = self.quats_raw
            self.quats_full, self.image_is_mirrored = collapse_to_half_space(self.quats_raw, plane_vec)
            self.pos_raw = quaternion_to_S2(self.quats_raw)
            self.pos_full = quaternion_to_S2(self.quats_full)

            # double the number of data points by augmentation
            df = np.concatenate((df, df))

            image_indices, bin_centers, occupancy, bin_ids = \
                bin_and_threshold(self.pos_full, params.ang_width, params.prd_thres_low, tessellator=params.tess_hemisphere_type, plane_vec=plane_vec)

            self.thres_low = params.prd_thres_low
            self.thres_high = params.prd_thres_high

            self.bin_centers = bin_centers
            self.defocus = df

            self.image_indices_full = image_indices
            self.thres_ids = bin_ids
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

            params.prd_n_active = len(self.thres_ids)

            params.save()
            self.save()


    def insert_anchor(self, id: int, anchor: Anchor):
        self.anchors[id] = anchor


    def remove_anchor(self, id: int):
        if id in self.anchors:
            self.anchors.pop(id)


    @property
    def anchor_ids(self):
        return sorted(list(self.anchors.keys()))


    @property
    def thresholded_image_indices(self):
        thres_images = self.image_indices_full[self.thres_ids]
        for i in range(thres_images.size):
            if len(thres_images[i]) > self.thres_high:
                thres_images[i] = thres_images[i][:self.thres_high]

        return thres_images


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
