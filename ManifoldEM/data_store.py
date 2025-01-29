import os

from enum import Enum
import mrcfile
import numbers
import numpy as np
import pickle
import h5py
from dataclasses import dataclass

from typing import List, Any, Tuple, Dict, Set
from nptyping import NDArray, Shape, Int, Int64, Float64, Bool
from scipy.ndimage import shift
from scipy.fftpack import fft2, ifft2

from ManifoldEM.params import params
from ManifoldEM.star import get_align_data
from ManifoldEM.quaternion import collapse_to_half_space, quaternion_to_S2
from ManifoldEM.S2tessellation import bin_and_threshold
from ManifoldEM.FindCCGraph import op as FindCCGraph
import ManifoldEM.myio as myio
from ManifoldEM.util import get_CTFs, rotate_fill


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
    def from_index(idx: int) -> "Sense":
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


@dataclass
class PrdInfo:
    """
    Data class to store information about a projection direction.

    Attributes
    ----------
    prd_index : int
        The index of the projection direction after thresholding.
    S2_bin_index : int
        The index of the bin on the unit sphere. This is the index before thresholding.
    bin_center : ndarray
        The coordinates of center of the bin on the unit sphere.
    occupancy : int
        The number of images in the bin of this projection direction.
    trash : bool
        Whether the projection direction is marked as trash. Trash projection directions are not used in analysis.
    anchor : bool
        Whether the projection direction is marked as an anchor node. Anchor nodes propogate 'sense' to cluster neighbors.
    cluster_id : int
        The cluster ID. Projection directions in one cluster use optical flow to detect 'senses' for neighboring prds in the same cluster.
    raw_image_indices : ndarray
        The `occupancy` indices of the raw images in the image stack file.
    image_centers : ndarray
        The `occupancy` centers of the images on the unit sphere.
    image_quats : ndarray
        The `occupancy` quaternions of the images representing rotations from the z-axis to their position and rotation on the unit sphere.
    image_rotations : ndarray
        The `occupancy` rotations of the images in the image stack file in degrees.
    image_mirrored : ndarray[bool]
        The `occupancy` boolean values indicating whether the image is mirrored (orientation opposite mirror plane).

    """

    prd_index: int
    S2_bin_index: int
    bin_center: NDArray[Shape["3"], Float64]
    occupancy: int
    trash: bool
    anchor: bool
    cluster_id: int
    raw_image_indices: NDArray[Shape["Any"], Int]
    image_offsets: NDArray[Shape["Any,2"], Float64]
    image_centers: NDArray[Shape["Any,3"], Int]
    image_quats: NDArray[Shape["Any,4"], Float64]
    image_rotations: NDArray[Shape["Any"], Float64]
    image_mirrored: NDArray[Shape["Any"], Bool]
    image_filter: NDArray[Shape["Any,Any"], Float64]
    image_mask: NDArray[Shape["Any,Any"], Float64]

    def __repr__(self):
        relevant_fields = [
            "prd_index",
            "S2_bin_index",
            "bin_center",
            "occupancy",
            "trash",
            "anchor",
            "cluster_id",
        ]
        info_str = "\n".join(
            [f"{field}: {getattr(self, field)}" for field in relevant_fields]
        )

        return info_str


class PrdData:
    """
    Represents a single projection direction, providing access to its raw and transformed images, CTF images, and metadata.

    Attributes
    ----------
    info : PrdInfo
        Metadata about the projection direction.
    raw_images : ndarray
        The raw images from the image stack associated with the projection direction.
    transformed_images : ndarray
        The filtered and "in-plane" rotated images associated with the projection direction.
    ctf_images : ndarray
        The Contrast Transfer Function (CTF) images associated with the projection direction.
    dist_data : dict
        The distance information between images in the projection direction, including transformed images
        in the `transformed_images` attribute, and the CTF images in the `ctf_images` attribute.
    psi_data : dict
        The embedding data associated with the projection direction.
    EL_data : dict
        The NLSA/eigenvalue data associated with the projection direction.
    """

    def __init__(self, prd_index: int):
        prds = data_store.get_prds()
        if prd_index >= prds.n_bins:
            raise ValueError("Invalid prd index")

        self._image_indices = prds.thresholded_image_indices[prd_index]
        self._dist_data = None
        self._raw_images = None
        self._transformed_images = None
        self._psi_data = None
        self._EL_data = None
        self._CTF = None

        with h5py.File(params.get_dist_file(prd_index), "r") as f:
            rotations = np.array(f["rotations"])
            mask = np.array(f["msk2"])
            image_filter = np.array(f["image_filter"])

        image_offsets = prds.microscope_origin
        image_offsets = np.empty((len(self._image_indices), 2))
        image_offsets[:, 0] = prds.microscope_origin[1][self._image_indices]
        image_offsets[:, 1] = prds.microscope_origin[0][self._image_indices]

        self._info = PrdInfo(
            prd_index=prd_index,
            S2_bin_index=prds.thres_ids[prd_index],
            bin_center=prds.bin_centers[:, prds.thres_ids[prd_index]],
            occupancy=prds.occupancy[prd_index],
            trash=prd_index in prds.trash_ids,
            anchor=prd_index in prds.anchor_ids,
            cluster_id=prds.cluster_ids[prd_index],
            raw_image_indices=self._image_indices,
            image_offsets=image_offsets,
            image_centers=prds.pos_full[:, self._image_indices].T,
            image_quats=prds.quats_full[:, self._image_indices].T,
            image_rotations=rotations,
            image_mirrored=prds.image_is_mirrored[self._image_indices],
            image_filter=image_filter,
            image_mask=mask,
        )

    def _load_psi_data(self):
        if self._psi_data is None:
            psi_file = params.get_psi_file(self._info.prd_index)
            if os.path.isfile(psi_file):
                self._psi_data = myio.fin1(psi_file)
            else:
                msg = f"Embedding data file not found: {psi_file}"
                raise FileNotFoundError(msg)

        return self._psi_data

    def _load_EL_data(self):
        if self._EL_data is None:
            EL_file = params.get_EL_file(self._info.prd_index)
            if os.path.isfile(EL_file):
                self._EL_data = myio.fin1(EL_file)
            else:
                msg = f"EL data file not found: {EL_file}"
                raise FileNotFoundError(msg)

        return self._EL_data

    def _load_dist_data(self):
        if self._dist_data is None:
            dist_file = params.get_dist_file(self._info.prd_index)
            if os.path.isfile(dist_file):
                self._dist_data = myio.fin1(dist_file)
            else:
                msg = f"Distance data file not found: {dist_file}"
                raise FileNotFoundError(msg)

        return self._dist_data

    @property
    def info(self):
        return self._info

    @property
    def psi_data(self):
        return self._load_psi_data()

    @property
    def EL_data(self):
        return self._load_EL_data()

    @property
    def dist_data(self):
        return self._load_dist_data()

    @property
    def raw_images(self):
        if self._raw_images is None:
            img_stack_data = data_store.get_image_stack_data()
            self._raw_images = np.empty(
                shape=(
                    len(self._image_indices),
                    params.ms_num_pixels,
                    params.ms_num_pixels,
                ),
                dtype=np.float32,
            )
            for i, idx in enumerate(self._image_indices):
                self._raw_images[i] = img_stack_data[idx]

        return self._raw_images

    @property
    def transformed_images(self):
        if self._transformed_images is None:
            img_stack_data = data_store.get_image_stack_data()
            imgAll = np.empty(
                shape=(
                    len(self._image_indices),
                    params.ms_num_pixels,
                    params.ms_num_pixels,
                ),
                dtype=np.float32,
            )

            for i, idx in enumerate(self._image_indices):
                imgAll[i] = img_stack_data[idx]

                imgAll[i] = shift(
                    imgAll[i],
                    (self.info.image_offsets[i, 0] - 0.5, self.info.image_offsets[i, 1] - 0.5),
                    order=3,
                    mode="wrap",
                )

                if self.info.image_mirrored[i]:
                    imgAll[i] = np.flipud(imgAll[i])

                imgAll[i] = ifft2(fft2(imgAll[i]) * self.info.image_filter).real
                imgAll[i] = rotate_fill(imgAll[i], self.info.image_rotations[i])
                imgAll[i] = imgAll[i] * self.info.image_mask

            self._transformed_images = imgAll

        return self._transformed_images

    @property
    def ctf_images(self):
        if self._CTF is None:
            prds = data_store.get_prds()
            self._CTF = get_CTFs(
                params.ms_num_pixels,
                prds.get_defocus_by_prd(self._info.prd_index),
                params.ms_spherical_aberration,
                params.ms_kilovolts,
                params.ms_ctf_envelope,
                params.ms_amplitude_contrast_ratio,
            )

        return self._CTF

    def __repr__(self) -> str:
        return self._info.__repr__()


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

    Properties
    ----------
    Provides access to various subsets or transformations of the data, such as occupancy without duplication,
    bin centers without duplication, anchor IDs, thresholded image indices, occupancy for thresholded IDs,
    the number of bins, and the number of thresholded entries.
    """

    def __init__(self):
        self.thres_low: int = params.prd_thres_low
        self.thres_high: int = params.prd_thres_high
        self.bin_centers: NDArray[Shape["3,Any"], Float64] = np.empty(shape=(3, 0))

        self.defocus: NDArray[Shape["Any"], Float64] = np.empty(0)
        self.microscope_origin: Tuple[
            NDArray[Shape["Any"], Float64], NDArray[Shape["Any"], Float64]
        ] = (np.empty(0), np.empty(0))

        self.pos_raw: NDArray[Shape["3,Any"], Float64] = np.empty(shape=(3, 0))
        self.pos_full: NDArray[Shape["3,Any"], Float64] = np.empty(shape=(3, 0))
        self.quats_raw: NDArray[Shape["4,Any"], Float64] = np.empty(shape=(4, 0))
        self.quats_full: NDArray[Shape["4,Any"], Float64] = np.empty(shape=(4, 0))

        self.image_indices_full: NDArray[Shape["Any"], Any] = np.empty(0, dtype=object)
        self.thres_ids: list[int] = []
        self.occupancy_full: NDArray[Shape["Any"], Int] = np.empty(0, dtype=int)

        self.anchors: Dict[int, Anchor] = {}
        self.trash_ids: Set[int] = set()
        self.reembed_ids: Set[int] = set()

        self.neighbor_graph: Dict[str, Any] = {}
        self.neighbor_subgraph: List[Dict[str, Any]] = []

        self.neighbor_graph_pruned: Dict[str, Any] = {}
        self.neighbor_subgraph_pruned: List[Dict[str, Any]] = []

        self.pos_thresholded: NDArray[Shape["3,Any"], Float64] = np.empty(shape=(3, 0))
        self.theta_thresholded: NDArray[Shape["Any"], Float64] = np.empty(0)
        self.phi_thresholded: NDArray[Shape["Any"], Float64] = np.empty(0)
        self.cluster_ids: NDArray[Shape["Any"], Int] = np.empty(0, dtype=int)

        self.image_is_mirrored: NDArray[Shape["Any"], Bool] = np.empty(0, dtype=bool)

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

        with open(pd_file, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save(self):
        """
        Saves the current projection direction metadata (this object) to a file.
        """
        with open(params.pd_file, "wb") as f:
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
        force_rebuild = bool(os.environ.get("MANIFOLD_REBUILD_DS", 0))
        if (
            force_rebuild
            or self.pos_full.size == 0
            or self.thres_low != params.prd_thres_low
            or self.thres_high != params.prd_thres_high
        ):
            if force_rebuild:
                print("Rebuilding data store")
                os.environ.pop("MANIFOLD_REBUILD_DS")

            print("Calculating projection direction information")
            self.microscope_origin, self.quats_raw, U, V = get_align_data(
                params.align_param_file, flip=True
            )
            df = (U + V) / 2

            plane_vec = np.array(params.tess_hemisphere_vec)
            self.quats_raw = self.quats_raw
            self.quats_full, self.image_is_mirrored = collapse_to_half_space(
                self.quats_raw, plane_vec
            )
            self.pos_raw = quaternion_to_S2(self.quats_raw)
            self.pos_full = quaternion_to_S2(self.quats_full)

            # double the number of data points by augmentation
            df = np.concatenate((df, df))

            image_indices, bin_centers, occupancy, bin_ids = bin_and_threshold(
                self.pos_full,
                params.ang_width,
                params.prd_thres_low,
                tessellator=params.tess_hemisphere_type,
                plane_vec=plane_vec,
            )

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
            self.phi_thresholded = (
                np.arctan2(self.pos_thresholded[1, :], self.pos_thresholded[0, :])
                * 180.0
                / np.pi
            )
            self.theta_thresholded = (
                np.arccos(self.pos_thresholded[2, :]) * 180.0 / np.pi
            )

            # double bins because routine expects number of bins on full sphere, while we bin
            # only half
            self.neighbor_graph, self.neighbor_subgraph = FindCCGraph(
                self.thresholded_image_indices, 2 * self.n_bins, self.pos_thresholded
            )

            def get_cluster_ids(G):
                nodesColor = np.zeros(G["nNodes"], dtype="int")
                for i, nodesCC in enumerate(G["NodesConnComp"]):
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

    def get_prd_data(self, id: int):
        """
        Returns a PrdData object for the given projection direction ID.

        Parameters
        ----------
        id : int
            The index of the projection direction.

        Raises
        ------
        ValueError
            If the ID is invalid.
        """
        if not isinstance(id, numbers.Integral):
            raise TypeError("Invalid prd index type")

        if id < 0 or id >= self.n_bins:
            msg = f"Invalid prd index: {id}. Valid indices on [0, {self.n_bins})"
            raise ValueError(msg)

        return PrdData(id)

    def get_defocus_by_prd(self, prd_index: int):
        """
        Returns the defocus value of the images in a given projection direction.

        Parameters
        ----------
        prd_index : int
            The index of the projection direction.

        Returns
        -------
        ndarray
            The defocus values for the images in the given projection direction.
        """
        return self.defocus[self.thresholded_image_indices[prd_index]]

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
                thres_images[i] = thres_images[i][: self.thres_high]

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
        if not hasattr(cls, "instance"):
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

    def get_prd_data(self, prd_index: int):
        self._projection_directions.update()
        return self._projection_directions.get_prd_data(prd_index)

    def get_image_stack_data(self):
        """
        Returns
        -------
        mrcfile.mmap
            Singular mrcfile.mmap().data instance for the image stack file.
        """

        if self._image_stack_data is None:
            self._image_stack_data = mrcfile.mmap(params.img_stack_file, "r").data

        return self._image_stack_data


data_store = _DataStore()
