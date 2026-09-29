from __future__ import annotations

import copy
import h5py
import numpy as np

from cv2 import calcOpticalFlowFarneback
from fasthog import hog_from_gradient as histogram_from_gradients
from functools import partial
from nptyping import Integer, NDArray, Float, Shape
from typing import Any, Union
from numpy import linalg as LA
import numpy.typing as npt
from ManifoldEM.data_store import H5Dataset, ProjectionDirections
from ManifoldEM.util import (
    recursive_dict_to_hdf5,
    dispatch_func,
    dispatch_helper,
)
from ManifoldEM.CC.hornschunck_simple import lowpassfilt, hornschunck_simple
from ManifoldEM.belief_propagation import belief_propagation, BeliefPropagationOptions


def get_orient_mag(X, Y):
    orient = np.arctan2(Y, X) % (2 * np.pi)
    mag = np.sqrt(X**2 + Y**2)
    return orient, mag


def rescale_linear(M, edgeNumRange, mvalrange):
    numE = max(edgeNumRange)
    nm = np.zeros(numE + 1, dtype=int)
    all_m = []

    for e in edgeNumRange:
        nm[e] = M[e].size
        all_m.append(M[e].ravel())

    all_m = np.squeeze(all_m).flatten()

    # determine if there are outliers in the all_m array
    q1, q3 = np.percentile(all_m, [25, 75])
    iqr = q3 - q1
    upper_thresh = q3 + (1.5 * iqr)

    all_m[all_m > upper_thresh] = upper_thresh

    # linear scaling of values within the range 'mvalr', min and max to mapped to min(mvalr) and max(mvalr)
    scaled_all_m = np.interp(all_m, (np.min(all_m), np.max(all_m)), mvalrange)
    M_scaled = np.empty(M.shape, dtype=object)
    nm_start = 0
    for e in edgeNumRange:
        nm_ind = np.arange(nm_start, nm_start + nm[e])
        nm_start += nm[e]
        M_scaled[e] = np.reshape(scaled_all_m[nm_ind], M[e].shape)

    return M_scaled


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0, 1.0)):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff3(stack, niter, kappa, gamma, step)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    stack = stack.astype("float32")
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    for _ in np.arange(1, niter):
        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!), Perona-Malik equation 1
        gD = np.exp(-((deltaD / kappa) ** 2.0)) / step[0]
        gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[1]
        gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

    return stackout


def optical_flow_movie(
    movie,
    blockSize_avg,
    label,
    InitialFlowVec: Union[None, dict] = None,
):
    # movie: (n_frames, n_pix, n_pix)
    num_frames, dim = movie.shape[:2]

    if not (label == "FWD" or label == "REV"):
        msg = f"Invalid label '{label}'. Must be either 'FWD' or 'REV'"
        raise ValueError(msg)

    VxM = np.zeros((dim, dim))
    VyM = np.zeros((dim, dim))

    if InitialFlowVec:
        FlowVec = copy.deepcopy(InitialFlowVec)
    else:
        FlowVec = {}

    if not InitialFlowVec:
        # Gunnar-Farneback flow as the initial estimate for Horn-Schunck, on low-pass filtered frames
        sig = 1.5  # sigma for lowpass gauss filter

        # use a median fitler for the optical flow vector field
        movie = anisodiff3(
            movie,
            niter=5,
            kappa=50,
            gamma=0.1,
            step=(5.0, 3.0, 3.0),
        )

        # average the movie
        numAvgFrames = np.ceil(float(num_frames) / blockSize_avg).astype(int)
        AvgMov = np.zeros((numAvgFrames, dim, dim))
        for b in range(0, numAvgFrames):
            frameStart = b * blockSize_avg
            frameEnd = min((b + 1) * blockSize_avg, num_frames)

            blockMovie = movie[frameStart:frameEnd, :, :]
            AvgMov[b, :, :] = np.mean(blockMovie, axis=0, dtype=np.float64)

        # start Optical flow algorithm
        ImgFrame_prev = lowpassfilt(AvgMov[0, :, :], sig)

        for frameno in range(0, numAvgFrames):
            ImgFrame_curr = lowpassfilt(AvgMov[frameno, :, :], sig)

            flow = calcOpticalFlowFarneback(
                ImgFrame_prev,
                ImgFrame_curr,
                flow=None,
                pyr_scale=0.4,
                levels=5,
                winsize=21,
                iterations=10,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
            Vx = flow[:, :, 0]
            Vy = flow[:, :, 1]

            # GF provides the initial estimates for HS
            Vx, Vy = hornschunck_simple(
                ImgFrame_prev, ImgFrame_curr, Vx, Vy, sig, 2.0, 200
            )

            VxM = VxM + Vx
            VyM = VyM + Vy

            ImgFrame_prev = np.copy(ImgFrame_curr)

        # store the flow vectors in a dictionary
        FlowVec = dict(
            Vx=VxM.astype(np.float16), Vy=VyM.astype(np.float16), Orient=[], Mag=[]
        )

    else:
        # read input FWD vectors when available
        VxM = FlowVec["Vx"]
        VyM = FlowVec["Vy"]

    # temporary trial of getting negative vectors directly from FW vectors
    if label == "REV":
        VxM = copy.deepcopy(-1.0 * VxM)
        VyM = copy.deepcopy(-1.0 * VyM)
        FlowVec["Vx"] = VxM.astype(np.float16)
        FlowVec["Vy"] = VyM.astype(np.float16)

    # get orientation and magnitude of the flow vectors
    FOrientMat, FMagMat = get_orient_mag(VxM, VyM)

    FlowVec["Orient"] = FOrientMat.astype(np.float16)
    FlowVec["Mag"] = FMagMat.astype(np.float16)

    return FlowVec


def compute_psi_movie_optical_flow(movie):
    blockSize_avg = 5  # how many frames will used for normal averaging
    movie_fwd = np.copy(movie)

    flow_vec_fwd = optical_flow_movie(movie_fwd, blockSize_avg, "FWD")
    # MFWD is used but due to label of 'REV', the negative vectors will be used after getting the FWD vectors
    flow_vec_rev = optical_flow_movie(movie_fwd, blockSize_avg, "REV", flow_vec_fwd)

    return dict(FWD=flow_vec_fwd, REV=flow_vec_rev)


def optical_flow(nlsa_movies: list[NDArray[Shape["Any,Any,Any"], Float]]):
    num_psi = len(nlsa_movies)
    # calculate OF for each psi-movie
    FlowVecPrD = [{}] * num_psi
    for psi_index in range(num_psi):
        # FIXME: Why is this negative?
        movie = -np.asarray(nlsa_movies[psi_index], dtype=np.float64)
        FlowVecPrD[psi_index] = compute_psi_movie_optical_flow(movie)

    return dict(FlowVecPrD=FlowVecPrD)


def optical_flow_movie_list(
    nlsa_movies: list[list[NDArray[Shape["Any,Any,Any"], Float]]] | list[list[H5Dataset]],
    ncpu: int = 1,
):
    return dispatch_func(
        partial(dispatch_helper, func=optical_flow),
        [dict(nlsa_movies=movies) for movies in nlsa_movies],
        "Computing Optical Flow",
        ncpu=ncpu,
    )


def validate_anchor_nodes(G, anchor_list, trash_ids, anchors, allow_empty_clusters):
    print(f"Number of anchor nodes: {len(anchor_list)}")
    print(f"Anchor list: {anchor_list}")
    if set(anchor_list).intersection(set(G["Nodes"])):
        if len(anchor_list) + len(trash_ids) == G["nNodes"]:
            print(
                "All nodes have been manually selected (as anchor nodes). "
                "Conformational-coordinate propagation is not required\n"
            )

            psinums = np.zeros((2, G["nNodes"]), dtype=int)
            senses = np.zeros((2, G["nNodes"]), dtype=int)

            for id, anchor in anchors.items():
                psinums[0, id] = anchor.CC - 1
                senses[0, id] = anchor.sense.value

            for trash_index in trash_ids:
                psinums[0, trash_index] = -1
                senses[0, trash_index] = 0

            return psinums, senses
    else:
        if not allow_empty_clusters:
            raise ValueError(
                "Missing anchor nodes. All clusters must have at least one anchor node"
            )

    return None, None


def get_cluster_nodes_edges(
    Gsub, anchor_ids: list[int], num_clusters: int, allow_empty_clusters: bool, trash_ids=frozenset()
):
    nodelCsel = []
    edgelCsel = []
    # this list keeps track of the connected component (single nodes included) for which no anchor was provided
    cluster_no_anchor = []
    anchor_id_set = set(anchor_ids)
    for i in range(num_clusters):
        nodesGsubi = set(Gsub[i]["originalNodes"])
        edgelistGsubi = Gsub[i]["originalEdgeList"]

        if anchor_id_set.intersection(nodesGsubi):
            nodelCsel.append(list(nodesGsubi))
            edgelCsel.append(edgelistGsubi[0])
        else:
            cluster_no_anchor.append(i)

    # pruning isolates every trashed PrD: those components need no anchor, and get no coordinate
    missing = [i for i in cluster_no_anchor if not set(Gsub[i]["originalNodes"]) <= set(trash_ids)]
    if missing:
        missing_prds = [sorted(int(n) for n in Gsub[i]["originalNodes"]) for i in missing]
        print(
            f"Warning: no anchor in connected components {missing}, with the PrDs (0-based) {missing_prds}."
        )
        if not allow_empty_clusters:
            raise ValueError(
                f"No anchor in the connected components with the PrDs (0-based) {missing_prds}. "
                "Anchor at least one PrD in each connected component, or run this routine again "
                "with allow_empty_clusters=True to leave those PrDs without a coordinate."
            )

    node_range = np.sort(np.concatenate(nodelCsel))
    edge_num_range = np.sort(np.concatenate(edgelCsel))

    return node_range, edge_num_range, cluster_no_anchor


def select_flow_vec(flow_vec, flow_vec_pct_thresh):
    # keep only the flow vectors whose magnitude is above the given percentile
    VxM, VyM = flow_vec["Vx"], flow_vec["Vy"]
    FMagMat = flow_vec["Mag"]
    magThresh = np.percentile(FMagMat.flatten(), flow_vec_pct_thresh)
    not_magThIdx = np.where(FMagMat <= magThresh)

    VxMSel, VyMSel = np.copy(VxM), np.copy(VyM)
    VxMSel[not_magThIdx] = 0.0
    VyMSel[not_magThIdx] = 0.0

    return dict(Vx=VxMSel, Vy=VyMSel)


def HOGOpticalFlow(flowVec):
    return histogram_from_gradients(
        flowVec["Vx"].astype(np.float64),
        flowVec["Vy"].astype(np.float64),
        cell_size=(4, 4),
        cells_per_block=(2, 2),
        n_bins=9,
        signed=True,
        norm_type="L2-Hys",
    )


def compare_orient_matrix(flow_vec_a, flow_vec_b):
    """How different two flow fields are: the l2 distance between their histograms of oriented gradients."""
    return LA.norm(HOGOpticalFlow(flow_vec_a) - HOGOpticalFlow(flow_vec_b))


def compute_edge_measure_pair_wise_all_psi(
    flow_vec_prd: list[dict[str, Any]],
    flow_vec_nbr_prd: list[dict[str, Any]],
    flow_vec_pct_thresh: float,
):
    """
    The (num_psi, 2 * num_psi) edge measure between a PrD and its neighbour: for each psi movie of the
    PrD played forward, its difference to each psi movie of the neighbour played forward, then reversed.
    """
    num_psi = len(flow_vec_prd)
    measure_fwd = np.empty((num_psi, num_psi))
    measure_rev = np.empty((num_psi, num_psi))
    for i_psi in range(num_psi):
        flow_vec_prd_fwd = select_flow_vec(
            flow_vec_prd[i_psi]["FWD"], flow_vec_pct_thresh
        )

        # psi selection candidates for the neighboring prD
        for psinum_neighb in range(num_psi):
            flow_vec_nbr_prd_fwd = select_flow_vec(
                flow_vec_nbr_prd[psinum_neighb]["FWD"], flow_vec_pct_thresh
            )
            flow_vec_nbr_prd_rev = select_flow_vec(
                flow_vec_nbr_prd[psinum_neighb]["REV"], flow_vec_pct_thresh
            )
            measure_fwd[i_psi, psinum_neighb] = compare_orient_matrix(flow_vec_prd_fwd, flow_vec_nbr_prd_fwd)
            measure_rev[i_psi, psinum_neighb] = compare_orient_matrix(flow_vec_prd_fwd, flow_vec_nbr_prd_rev)

    return np.hstack((measure_fwd, measure_rev))


def compute_edge_measures_all(
    edges: list[tuple[int, int]],
    edge_nums: list[int] | npt.NDArray[np.int_],
    flow_map: dict[int, dict[str, Any]],
    flow_vec_pct_thresh: float,
    ncpu: int = 1,
):
    input_data = [
        dict(
            flow_vec_prd=flow_map[edges[edge_num][0]]["FlowVecPrD"],
            flow_vec_nbr_prd=flow_map[edges[edge_num][1]]["FlowVecPrD"],
            flow_vec_pct_thresh=flow_vec_pct_thresh,
        )
        for edge_num in edge_nums
    ]

    edge_measures_by_position = dispatch_func(
        partial(dispatch_helper, func=compute_edge_measure_pair_wise_all_psi),
        input_data,
        "Computing Edge Measures",
        ncpu=ncpu,
    )

    # dispatch_func numbers the results by their position in edge_nums
    edge_measures = np.empty(len(edges), dtype=object)
    for position, e in enumerate(edge_nums):
        edge_measures[e] = edge_measures_by_position[position]

    # This rescaling step is to prevent underflow/overflow, should be checked if does not work
    scale_range = [5, 45]
    edge_measures = rescale_linear(edge_measures, edge_nums, scale_range)

    return edge_measures


def calculate_probability_landscape(
    psinums: NDArray[Shape["Any"], Integer],
    senses: NDArray[Shape["Any"], Integer],
    taus: list[NDArray[Shape["Any"], Float]] | list[h5py.Dataset],
    states_per_coord: int,
):
    """Occupancy of the `states_per_coord` states along the conformational coordinate, and the
    per-PrD normalised taus (sense applied) that it is the histogram of."""
    occupancy = np.zeros((1, states_per_coord)).flatten()
    tau_avg = np.array([])

    for prd in range(len(psinums)):
        psi = psinums[prd]
        tau = np.array(taus[prd][psi]).flatten()
        if senses[prd] == -1:
            tau = 1 - tau

        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        h, _ = np.histogram(tau, states_per_coord)
        occupancy = occupancy + h
        tau_avg = np.concatenate((tau_avg, tau.flatten()))

    return occupancy, tau_avg


def write_conformational_coords(
    output_file: str,
    psinums: NDArray[Shape["Any"], Integer],
    senses: NDArray[Shape["Any"], Integer],
    flow_data: None | dict[int, dict[str, Any]] = None,
):
    """
    Writes every PrD's psinum and sense (-1 and 0 for the PrDs without a coordinate) and the flow
    data of the PrDs in `flow_data`, and removes the flow data of the other PrDs, which is from an
    earlier run. Opened for writing only now: while it is open for writing, the optical-flow workers
    couldn't read it.
    """
    with h5py.File(output_file, "a") as output_handle:
        for index in range(len(psinums)):
            group = output_handle.get(f"prd_{index}")
            if not isinstance(group, h5py.Group):
                msg = f"path 'prd_{index}' is not a group in the output handle"
                raise ValueError(msg)

            if "sense" in group:
                group["sense"][...] = senses[index]
            else:
                group["sense"] = senses[index]
            if "psinum" in group:
                group["psinum"][...] = psinums[index]
            else:
                group["psinum"] = psinums[index]

        flow_data = flow_data or {}
        for index in range(len(psinums)):
            group = output_handle[f"prd_{index}"]
            if "flow_data" in group:
                del group["flow_data"]
            if index in flow_data:
                recursive_dict_to_hdf5(group.create_group("flow_data"), flow_data[index])


def find_conformational_coords(
    prds: ProjectionDirections,
    nlsa_movies: list[list[NDArray[Shape["Any,Any,Any"], Float]]] | list[list[H5Dataset]],
    num_psi: int,
    output_file: None | str = None,
    flow_vec_pct_thresh: float = 95,
    allow_empty_clusters: bool = False,
    ncpu: int = 1,
):
    """
    Finds each PrD's conformational coordinate (psinum, and sense +1/-1) by belief propagation from
    the anchors, and returns them (-1 and 0 for the PrDs without one). With `output_file`, writes them
    and the flow data to that prd_analysis.h5.
    """
    G, Gsub = prds.prune_graphs()
    num_clusters = len(G["NodesConnComp"])
    anchor_list = prds.anchor_ids

    psinums, senses = validate_anchor_nodes(
        G, anchor_list, prds.trash_ids, prds.anchors, allow_empty_clusters
    )
    if psinums is not None:
        if output_file:
            write_conformational_coords(output_file, psinums[0], senses[0])
        return psinums[0], senses[0]

    node_range, edge_num_range, cluster_no_anchor = get_cluster_nodes_edges(
        Gsub, anchor_list, num_clusters, allow_empty_clusters, prds.trash_ids
    )
    G.update(ConnCompNoAnchor=cluster_no_anchor)

    # optical flow only for the PrDs of the anchored components, as main did: the others get no
    # edge measures and no coordinate
    flow_prds = [int(i) for i in node_range]
    flow_list = optical_flow_movie_list([nlsa_movies[i] for i in flow_prds], ncpu)
    flow_map = {prd: flow_list[k] for k, prd in enumerate(flow_prds)}

    edge_measures = compute_edge_measures_all(
        G["Edges"], edge_num_range, flow_map, flow_vec_pct_thresh, ncpu
    )

    psinums, senses, node_belief = belief_propagation(
        prds, num_psi, G, BeliefPropagationOptions(), edge_measures
    )

    if output_file:
        edges = G["Edges"]
        flow_data = {}
        for index in flow_map.keys():
            edge_measure_indices = np.where(
                (edges[:, 0] == index) | (edges[:, 1] == index)
            )[0]
            edge_measures_local = dict()
            for edge_index in edge_measure_indices:
                i = edges[edge_index][0]
                if i == index:
                    i = edges[edge_index][1]

                edge_measures_local[i] = edge_measures[edge_index]

            flow_data[index] = dict(
                flow_map=flow_map[index],
                node_belief=node_belief[:, index],
                edge_measures=edge_measures_local,
            )

        write_conformational_coords(output_file, psinums, senses, flow_data)

    return psinums, senses
