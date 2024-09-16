import os

from enum import Enum
import mrcfile
import numpy as np
import pickle

from typing import List, Any, Tuple, Dict, Set
from nptyping import NDArray, Shape, Int, Int64, Float64, Bool

from ManifoldEM.params import params
from ManifoldEM.star import get_align_data
from ManifoldEM.quaternion import collapse_to_half_space, quaternion_to_S2
from ManifoldEM.S2tessellation import bin_and_threshold
from ManifoldEM.FindCCGraph import op as FindCCGraph


class PrdData:
    def __init__(self, prd_index: int):
        prds = data_store.get_prds()
        if prd_index >= prds.n_bins:
            raise ValueError("Invalid prd index")

        self.prd_index = prd_index
        self.image_indices = prds.thresholded_image_indices[prd_index]
        self._dist_data = None
        self._raw_images = None

        dist_file = params.get_dist_file(prd_index)
        if os.path.isfile(dist_file):
            with open(dist_file, 'rb') as f:
                self._dist_data = pickle.load(f)

        if self.image_indices.size:
            img_stack_data = data_store.get_image_stack_data()
            self._raw_images = np.empty(shape=(len(self.image_indices), params.ms_num_pixels, params.ms_num_pixels), dtype=np.float32)
            for i, idx in enumerate(self.image_indices):
                self._raw_images[i] = img_stack_data[idx]


    @property
    def raw_images(self):
        return self._raw_images


    @property
    def transformed_images(self):
        return self._dist_data['imgAll']


    @property
    def ctf_images(self):
        return self._dist_data['CTF'].reshape(-1, params.ms_num_pixels, params.ms_num_pixels)


class Sense(Enum):
    """
    An enumeration to represent the direction of projection or alignment.

    Attributes
    ----------
    FWD : int
        Represents the forward direction.
    REV : int
        Represents the reverse direction.

    Methods
    -------
    from_index(idx: int) -> 'Sense'
        Static method that converts an integer index to a Sense enum.
    to_index() -> int
        Converts the Sense enum to an integer index.
    """
    FWD = 1
    REV = -1

    @staticmethod
    def from_index(idx: int) -> 'Sense':
        """
        Converts an integer index to a Sense enum.

        Parameters
        ----------
        idx : int
            The index to convert.

        Returns
        -------
        Sense
            The corresponding Sense enum value.

        Raises
        ------
        ValueError
            Raised if the index is invalid.
        """
        if idx == 0:
            return Sense.FWD
        if idx == 1:
            return Sense.REV
        raise ValueError("Invalid index")

    def to_index(self) -> int:
        """
        Converts the Sense enum to an integer index.

        Returns
        -------
        int
            The corresponding index value (0: FWD or 1: REV).
        """
        return 0 if self == Sense.FWD else 1


class Anchor:
    """
    Represents an anchor point with associated properties.

    Parameters
    ----------
    CC : int
        Conformational coordinate, defaulting to `1`.
    sense : Sense
        The direction of the projection, defaulting to `Sense.FWD`.

    Attributes
    ----------
    CC : int
        Conformational coordinate, defaulting to `1`.
    sense : Sense
        Direction of the projection.
    """
    def __init__(self, CC: int = 1, sense: Sense = Sense.FWD):
        self.CC: int = CC
        self.sense: Sense = sense


class _ProjectionDirections:
    """
    Manages and processes projection direction data, including thresholds, bin centers, defocus values, and more.

    Attributes
    ----------
    Various attributes to store thresholds, bin centers, defocus values, microscope origins, full position data,
    quaternion data, image indices, threshold IDs, occupancy data, anchor points, and neighbor graph data.

    Methods
    -------
    load(pd_file=None)
        Loads projection direction data from a file.
    save()
        Saves the current projection direction data to a file.
    update()
        Updates the projection direction data based on external parameters or files.
    insert_anchor(id: int, anchor: Anchor)
        Inserts an anchor point into the dataset.
    remove_anchor(id: int)
        Removes an anchor point from the dataset.
    deduplicate(arr)
        Removes duplicate entries from an array.

    Properties
    ----------
    Provides access to various subsets or transformations of the data, such as occupancy without duplication,
    bin centers without duplication, anchor IDs, thresholded image indices, occupancy for thresholded IDs,
    the number of bins, and the number of thresholded entries.
    """
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
        """
        Loads projection direction data from a specified file.

        Parameters
        ----------
        pd_file : Union[str,None], default=None
            The path to the file from which to load data. If `None`, uses the current loaded `params.pd_file`.
        """
        if pd_file is None:
            pd_file = params.pd_file

        with open(pd_file, 'rb') as f:
            self.__dict__.update(pickle.load(f))


    def save(self):
        """
        Saves the current projection direction metadata (this object) to a file.
        """
        with open(params.pd_file, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)


    def update(self):
        """
        Updates/builds the projection direction data based on external parameters or
        files. If data is uninitialized, the threshold values have changed, or the environment
        variable `MANIFOLD_REBUILD_DS` is set, the data is rebuilt. Otherwise, the data is just
        loaded from cache.
        """
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
        """
        Tags a given projection direction as an 'anchor'.

        Parameters
        ----------
        id : int
            The projection direction index for the anchor point.
        anchor : Anchor
            The `Anchor` data to associate with `id`.
        """
        self.anchors[id] = anchor


    def remove_anchor(self, id: int):
        """
        Untags an anchor point in the dataset. Does nothing if the anchor point does not exist.

        Parameters
        ----------
        id : int
            The identifier for the anchor point to remove.
        """
        if id in self.anchors:
            self.anchors.pop(id)


    @property
    def anchor_ids(self):
        """
        Get a sorted list of anchor IDs.

        Returns
        -------
        List[int]
            The sorted list of anchor IDs.
        """
        return sorted(list(self.anchors.keys()))


    @property
    def thresholded_image_indices(self):
        """
        Returns the image indices for thresholded data, applying the high threshold limit. I.e. if
        the number of images in a bin exceeds the high threshold, only the first `thres_high` image indices
        are returned for that bin.

        Returns
        -------
        ndarray[int]
            The thresholded image indices.
        """
        thres_images = self.image_indices_full[self.thres_ids]
        for i in range(thres_images.size):
            if len(thres_images[i]) > self.thres_high:
                thres_images[i] = thres_images[i][:self.thres_high]

        return thres_images


    @property
    def occupancy(self):
        """
        Returns array of the total number of images for each occupied projection direction.

        Returns
        -------
        ndarray[int]
            The occupancy data for thresholded IDs.
        """
        return self.occupancy_full[self.thres_ids]


    @property
    def n_bins(self):
        """
        Returns the number of bins on the S2 sphere.

        Returns
        -------
        int
            Number of bins on S2.
        """
        return self.bin_centers.shape[1]


    @property
    def n_thresholded(self):
        """
        Returns the number of thresholded entries.

        Returns
        -------
        int
            The number of thresholded entries.
        """
        return len(self.thres_ids)


class _DataStore:
    """
    Implements the Singleton design pattern to ensure a single instance of the data store.
    This class acts as a manager for _ProjectionDirections, providing global access to projection
    direction data and functionalities.

    Methods
    -------
    get_prds()
        Updates and returns the _ProjectionDirections instance, ensuring the data is current.
    get_image_stack_data()
        Returns the image stack data from the associated project mrcs file.
    """
    _projection_directions = _ProjectionDirections()
    _image_stack_data = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(_DataStore, cls).__new__(cls)
        return cls.instance


    def get_prds(self):
        """
        Returns
        -------
        _ProjectionDirections
            Singular _ProjectionDirections instance, updating the cache if necessary.
        """
        self._projection_directions.update()
        return self._projection_directions


    def get_image_stack_data(self):
        """
        Returns
        -------
        mrcfile.mmap
            Singular mrcfile.mmap().data instance for the image stack file.
        """

        if self._image_stack_data is None:
            self._image_stack_data = mrcfile.mmap(params.img_stack_file, 'r').data

        return self._image_stack_data


data_store = _DataStore()
