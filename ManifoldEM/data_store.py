import os

import numpy as np
import pickle

from typing import List, Any, Tuple, Dict
from nptyping import NDArray, Shape

from ManifoldEM.params import p
from ManifoldEM.Data import get_from_relion
from ManifoldEM.util import augment
from ManifoldEM.S2tessellation import op as tesselate

class _ProjectionDirections:
    def __init__(self):
        self.thres_low: int = p.PDsizeThL
        self.thres_high: int = p.PDsizeThH
        self.bin_centers: NDArray[Shape["3", Any], np.float64] = np.empty(shape=(3, 0))

        self.defocus: NDArray[np.float64] = np.empty(0)
        self.microscope_origin: Tuple[NDArray[np.float64], NDArray[np.float64]] = (np.empty(0), np.empty(0))

        self.pos_full: NDArray[Shape["3", Any], np.float64] = np.empty(shape=(3,0))
        self.quats_full: NDArray[Shape["4", Any], np.float64] = np.empty(shape=(4,0))

        self.pd_image_indices: NDArray[List[int]] = np.empty(0, dtype=object)
        self.thres_ids: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.occupancy_count: NDArray[int] = np.empty(0, dtype=int)

        self.anchor_ids: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.trash_ids: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        self.neighbor_graph: Dict[str, Any] = {}
        self.neighbor_subgraph: List[Dict[str, Any]] = []

        self.pos_thresholded: NDArray[Shape["3", Any], np.float64] = np.empty(shape=(3,0))
        self.theta_thresholded: NDArray[np.float64] = np.empty(0)
        self.phi_thresholded: NDArray[np.float64] = np.empty(0)
        self.cluster_ids: NDArray[int] = np.empty(0, dtype=int)

    def load(self, pd_file):
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
        if self.pos_full.size == 0 or self.thres_low != p.PDsizeThL or self.thres_high != p.PDsizeThH:
            print("Calculating projection direction information")
            sh, q, U, V = get_from_relion(p.align_param_file, flip=True)
            df = (U + V) / 2

            # double the number of data points by augmentation
            q = augment(q)
            df = np.concatenate((df, df))

            CG1, _, _, S2, _, S20, NC, NIND = tesselate(q, p.ang_width, p.PDsizeThL, False, p.PDsizeThH)

            self.thres_low = p.PDsizeThL
            self.thres_high = p.PDsizeThH

            self.bin_centers = S20
            self.defocus = df
            self.microscope_origin = sh

            self.pos_full = S2
            self.quats_full = q

            self.pd_image_indices = CG1
            self.thres_ids = NIND
            self.occupancy_count = NC

            self.anchor_ids = np.empty(0, dtype=np.int64)
            self.trash_ids = np.empty(0, dtype=np.int64)

            self.pos_thresholded = self.bin_centers[:, self.thres_ids]
            self.phi_thresholded = np.arctan2(self.pos_thresholded[1, :], self.pos_thresholded[0, :]) * 180. / np.pi
            self.theta_thresholded = np.arccos(self.pos_thresholded[2, :]) * 180. / np.pi

            from .FindCCGraph import op as FindCCGraph
            self.neighbor_graph, self.neighbor_subgraph = \
                FindCCGraph(self.thresholded_image_indices, self.n_bins, self.bin_centers[:, self.thres_ids])

            def get_cluster_ids(G):
                nodesColor = np.zeros(G['nNodes'], dtype='int')
                for i, nodesCC in enumerate(G['NodesConnComp']):
                    nodesColor[nodesCC] = i

                return nodesColor

            self.cluster_ids = get_cluster_ids(self.neighbor_graph)

            p.numberofJobs = len(self.thres_ids)

            p.save()
            self.save()


    @property
    def thresholded_image_indices(self):
        return self.pd_image_indices[self.thres_ids].copy()


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
