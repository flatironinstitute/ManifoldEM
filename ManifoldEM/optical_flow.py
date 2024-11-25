import copy
import h5py
import multiprocessing
import math
import numpy as np
import warnings

from cv2 import calcOpticalFlowFarneback
from fasthog import hog_from_gradient as histogram_from_gradients
from functools import partial
from nptyping import Integer, NDArray, Float, Shape
from typing import Any, Union
from numpy import linalg as LA
import numpy.typing as npt
from ManifoldEM.data_store import ProjectionDirections
from ManifoldEM.util import NullEmitter, get_tqdm, recursive_dict_to_hdf5
from ManifoldEM.CC.hornschunck_simple import lowpassfilt, op as hornschunk_simple
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


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0, 1.0), option=1):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

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

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

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

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-((deltaD / kappa) ** 2.0)) / step[0]
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[1]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2.0) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[2]

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


def find_bad_node_psi_tau(tau, tau_occ_thresh=0.33):
    quartile_1, quartile_3 = np.percentile(tau, [25, 75])
    iqr = quartile_3 - quartile_1

    ## Sept 2021
    # check if the tau value distribution have more than one narrow ranges far apart
    # this will artifically make the IQR value high giving the illusion of a wide tau
    # distribution. This cases need to be checked and set the IQR to a low value=0.01
    taubins = 50
    tau_h, _ = np.histogram(tau, bins=taubins)  # there are 50 states
    tau_h = np.array(tau_h)
    tau_nz = np.where(tau_h > 0.0)[0].size
    tau_occ = tau_nz / float(taubins)

    # if number of states present in tau values is less than occ_thresh=30%?then there are lot of
    # missing states,
    # tau_occ_thresh = 0.35 # 35% conservative here? # could input through p.py / gui ?
    badPsi = int((iqr < 0.02) or (tau_occ <= tau_occ_thresh))

    return badPsi, iqr, tau_occ


def load_prd_psi_movies_masked(
    nlsa_movies: list[NDArray[Shape["Any,Any"], Float]] | list[h5py.Dataset],
    taus: list[NDArray[Shape["Any"], Float]] | list[h5py.Dataset],
    mask: None | NDArray[Shape["Any,Any"], Float],
    find_bad_psi_tau: bool,
    tau_occ_thresh: float,
):
    num_psi = len(nlsa_movies)
    movie_prd_psis = [np.empty(0)] * num_psi
    tau_prd_psis = [np.empty(0)] * num_psi
    bad_psis = []
    tau_psis_iqr = []
    tau_psis_occ = []

    if len(nlsa_movies[0].shape) == 3:
        n_pix = nlsa_movies[0].shape[1]
    elif len(nlsa_movies[0].shape) == 2:
        n_pix = math.isqrt(nlsa_movies[0].shape[0])
        if n_pix * n_pix != nlsa_movies[0].shape[0]:
            raise ValueError(
                "Invalid movie shape. Dimension 0 must be a perfect square"
            )
    else:
        raise ValueError("Invalid movie list")

    if mask is None:
        mask = np.ones((n_pix, n_pix))

    for psinum in range(num_psi):
        IMG1 = np.array(nlsa_movies[psinum])
        tau = np.array(taus[psinum])

        # FIXME: Why is this negative?
        movie_prd_psis[psinum] = -IMG1 * mask
        tau_prd_psis[psinum] = tau

        if find_bad_psi_tau:
            b, tauIQR, tauOcc = find_bad_node_psi_tau(tau, tau_occ_thresh)
            tau_psis_iqr.append(tauIQR)
            tau_psis_occ.append(tauOcc)
            if b:
                bad_psis.append(psinum)
        else:
            bad_psis = []

    return movie_prd_psis, bad_psis, tau_prd_psis, tau_psis_iqr, tau_psis_occ


def optical_flow_movie(
    movie,
    blockSize_avg,
    label,
    InitialFlowVec: Union[None, dict] = None,
):
    # for display
    num_frames = movie.shape[0]
    if len(movie.shape) == 3:
        dim = movie.shape[1]
    elif len(movie.shape) == 2:
        dim = int(np.sqrt(movie.shape[1]))
    else:
        raise ValueError("Invalid movie shape")

    movie = np.resize(movie, (num_frames, dim, dim))

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
        do_filterImage = False
        sig = 2.0  # sigma for lowpass gauss filter
        OF_Type = "GF-HS"  # GF for initial estimates and then HS

        # only print for the first movie
        if OF_Type == "GF" or OF_Type == "GF-HS":
            do_filterImage = True
            if OF_Type == "GF-HS":
                sig = 1.5

        # use a median fitler for the optical flow vector field
        movie = anisodiff3(
            movie,
            niter=5,
            kappa=50,
            gamma=0.1,
            step=(5.0, 3.0, 3.0),
            option=1,
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
        ImgFrame_prev = AvgMov[0, :, :]
        if do_filterImage:
            ImgFrame_prev = lowpassfilt(ImgFrame_prev, sig)

        for frameno in range(0, numAvgFrames):
            ImgFrame_curr = AvgMov[frameno, :, :]

            if do_filterImage:
                ImgFrame_curr = lowpassfilt(ImgFrame_curr, sig)

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

            uInit = Vx  # if both methods are used , GF will provide intial estimates
            vInit = Vy  # if both methods are used , GF will provide intial estimates

            Vx, Vy = hornschunk_simple(
                ImgFrame_prev, ImgFrame_curr, uInit, vInit, sig, 2.0, 200
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


def optical_flow(
    nlsa_movies: list[NDArray[Shape["Any,Any"], Float]],
    taus: list[NDArray[Shape["Any"], Float]],
    mask: None | NDArray[Shape["Any,Any"], Float] = None,
    find_bad_psi_tau: bool = True,
    tau_occ_thresh: float = 0.33,
):
    # load movie and tau param first
    (
        movie_prd_psi,
        badNodesPsisTau,
        NodesPsisTauVals,
        NodesPsisTauIQR,
        NodesPsisTauOcc,
    ) = load_prd_psi_movies_masked(
        nlsa_movies, taus, mask, find_bad_psi_tau, tau_occ_thresh
    )

    num_psi = len(nlsa_movies)
    # calculate OF for each psi-movie
    FlowVecPrD = [{}] * num_psi
    for psi_index in range(num_psi):
        FlowVecPrD[psi_index] = compute_psi_movie_optical_flow(
            movie_prd_psi[psi_index],
        )

    return dict(
        FlowVecPrD=FlowVecPrD,
        badNodesPsisTau=np.array(badNodesPsisTau),
        NodesPsisTauIQR=NodesPsisTauIQR,
        NodesPsisTauOcc=NodesPsisTauOcc,
        NodesPsisTauVals=NodesPsisTauVals,
    )


def dispatch_func(
    func,
    input_data: list[Any],
    desc: str = "",
    ncpu: int = 1,
    progress_bar=NullEmitter(),
    progress_bounds=(0, 100),
):
    tqdm = get_tqdm()
    progress_min, progress_max = progress_bounds
    data: dict[int, Any] = {}
    if ncpu == 1:
        for i in tqdm(
            range(len(input_data)),
            desc=desc,
        ):
            data[i] = func(input_data[i])
            progress = int(
                progress_min + i / len(input_data) * (progress_max - progress_min)
            )
            progress_bar.emit(progress)
    else:
        with multiprocessing.Pool(ncpu) as pool:
            for i, result in tqdm(
                enumerate(pool.imap(func, input_data)),
                total=len(input_data),
                desc=desc,
            ):
                data[i] = result
                progress = int(
                    progress_min + i / len(input_data) * (progress_max - progress_min)
                )
                progress_bar.emit(progress)

    return data


def dispatch_helper(kwargs, func):
    return func(**kwargs)


def optical_flow_movie_list(
    nlsa_movies: list[list[NDArray[Shape["Any,Any"], Float]]]
    | list[list[h5py.Dataset]],
    taus: list[list[NDArray[Shape["Any"], Float]]] | list[list[h5py.Dataset]],
    mask: None | list[NDArray[Shape["Any,Any"], Float]],
    find_bad_psi_tau: bool,
    ncpu: int = 1,
    progress_bar=NullEmitter(),
    progress_bounds=(0, 100),
):
    input_data = [
        dict(
            nlsa_movies=nlsa_movies[i],
            taus=taus[i],
            mask=mask[i] if mask else None,
            find_bad_psi_tau=find_bad_psi_tau,
        )
        for i in range(len(nlsa_movies))
    ]

    return dispatch_func(
        partial(dispatch_helper, func=optical_flow),
        input_data,
        "Computing Optical Flow",
        ncpu=ncpu,
        progress_bar=progress_bar,
        progress_bounds=progress_bounds,
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
    Gsub, anchor_ids: list[int], num_clusters: int, allow_empty_clusters: bool
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

    if cluster_no_anchor:
        print(
            f"Warning: anchor node(s) in connected components {cluster_no_anchor} NOT selected."
        )
        if not allow_empty_clusters:
            raise ValueError(
                "Some (or all) of the anchor nodes are NOT in the Graph node list. "
                "Run this routine again with allow_empty_clusters=True or with at least "
                "one anchor in each connected component to proceed."
            )

    node_range = np.sort(np.concatenate(nodelCsel))
    edge_num_range = np.sort(np.concatenate(edgelCsel))

    return node_range, edge_num_range, cluster_no_anchor


def select_flow_vec(flow_vec, flow_vec_pct_thresh):
    # this will work for all elements for any ndarray [m x m x d],
    # Here for half-split movies we have d = 2, i.e. stack of two m x m matrix
    VxM, VyM = flow_vec["Vx"], flow_vec["Vy"]
    FMagMat = flow_vec["Mag"]
    magThresh = np.percentile(FMagMat.flatten(), flow_vec_pct_thresh)
    not_magThIdx = np.where(FMagMat <= magThresh)

    FMagSel = np.copy(FMagMat)
    FMagSel[not_magThIdx] = 0.0

    FOrientSel = np.copy(flow_vec["Orient"])
    FOrientSel[not_magThIdx] = -np.Inf

    # selecting which flow vectors to keep for analysis
    VxMSel, VyMSel = np.copy(VxM), np.copy(VyM)
    VxMSel[not_magThIdx] = 0.0
    VyMSel[not_magThIdx] = 0.0

    return dict(Vx=VxMSel, Vy=VyMSel, Orient=FOrientSel, Mag=FMagSel)


def HOGOpticalFlow(flowVec):
    cell_size = (4, 4)
    cells_per_block = (2, 2)
    n_bins = 9
    signed_orientation = True
    norm_type = "L2-Hys"

    hog_params = dict(
        cell_size=cell_size, cells_per_block=cells_per_block, n_bins=n_bins
    )
    VxDim = flowVec["Vx"].shape
    if len(VxDim) > 2:
        VxStackDim = VxDim[2]

        tempH = []
        for d in range(0, VxStackDim):
            gx = flowVec["Vx"][:, :, d].astype(np.float64)
            gy = flowVec["Vy"][:, :, d].astype(np.float64)

            tH = histogram_from_gradients(
                gx,
                gy,
                cell_size=cell_size,
                cells_per_block=cells_per_block,
                n_bins=n_bins,
                signed=signed_orientation,
                norm_type=norm_type,
            )
            tempH.append(tH)

        H = np.array(tempH)
        dims = np.shape(H)
        if len(dims) > 3:
            H = np.moveaxis(H, 0, -1)
    else:
        gx = flowVec["Vx"].astype(np.float64)
        gy = flowVec["Vy"].astype(np.float64)

        H = histogram_from_gradients(
            gx,
            gy,
            cell_size=cell_size,
            cells_per_block=cells_per_block,
            n_bins=n_bins,
            signed=signed_orientation,
            norm_type=norm_type,
        )

    return H, hog_params


# Compare how similar two Matrices/Images are.
# TODO: Implement error checking for wrong or, improper inputs
# Check for NaN or Inf outputs , etc.
def compare_orient_matrix(flow_vec_a, flow_vec_b, norm_type="l2"):
    if norm_type not in ["l1", "l2"]:
        raise ValueError("Invalid norm_type. Must be either 'l1' or 'l2'")

    HOGFA, hog_params = HOGOpticalFlow(flow_vec_a)
    HOGFB, hog_params = HOGOpticalFlow(flow_vec_b)

    # The dimensions of HOGFA and HOGFB should always match given the number of movie blocks created for movie A and B
    # if for some reason the number of blocks for movie A and B are different, then this check is a fail safe to make
    # the code still work
    hogDimA = HOGFA.shape

    hoffset = 1.25
    distHOGAB = []
    distHOGAB_tblock = []
    isBadPsiAB_block = []
    hp = np.ceil(hogDimA[0] / hog_params["cell_size"][0]).astype(int)
    num_hogel_th = np.ceil(0.2 * (hp**2) * hogDimA[2]).astype(int)

    if len(hogDimA) > 3:
        distHOGAB_tblock = np.zeros((hogDimA[3], 1))
        isBadPsiA_block = np.zeros((hogDimA[3], 1))
        isBadPsiB_block = np.zeros((hogDimA[3], 1))

        for j in range(hogDimA[3]):
            if np.count_nonzero(HOGFA[:, :, :, j]) <= num_hogel_th:
                HOGFA[:, :, :, j] = (
                    np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                )
                isBadPsiA_block[j] = 1

            if np.count_nonzero(HOGFB[:, :, :, j]) <= num_hogel_th:
                HOGFB[:, :, :, j] = (
                    np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                )
                isBadPsiB_block[j] = 1

            if norm_type == "l1":
                distHOGAB_tblock[j] = np.sum(
                    np.abs(HOGFA[:, :, :, j] - HOGFB[:, :, :, j])
                )
            elif norm_type == "l2":
                distHOGAB_tblock[j] = LA.norm(HOGFA[:, :, :, j] - HOGFB[:, :, :, j])

        isBadPsiAB_block = [isBadPsiA_block.T, isBadPsiB_block.T]

    # this should be done after the adjustments of the zero matrix to a matrix with high random numbers
    if norm_type == "l1":
        distHOGAB = np.sum(np.abs(HOGFA - HOGFB))
    elif norm_type == "l2":
        distHOGAB = LA.norm(HOGFA - HOGFB)

    return [distHOGAB, distHOGAB_tblock, isBadPsiAB_block]


def compare_psi_movies_optical_flow(FlowVecSelA, FlowVecSelB):
    # Analysis of the flow matrix
    psiMovFlowOrientMeasures = dict(Values=[], Values_tblock=[])
    Values, Values_tblock, isBadPsiAB_block = compare_orient_matrix(
        FlowVecSelA, FlowVecSelB
    )
    psiMovFlowOrientMeasures.update(Values=Values, Values_tblock=Values_tblock)

    return psiMovFlowOrientMeasures, isBadPsiAB_block


def compute_measures_psi_movies_optical_flow(
    FlowVecSelAFWD, FlowVecSelBFWD, FlowVecSelBREV
):
    psiMovOFMeasuresFWD, isBadPsiAB_blockF = compare_psi_movies_optical_flow(
        FlowVecSelAFWD, FlowVecSelBFWD
    )
    psiMovMFWD = psiMovOFMeasuresFWD["Values"]
    psiMovMFWD_tblock = psiMovOFMeasuresFWD["Values_tblock"]

    psiMovOFMeasuresREV, isBadPsiAB_blockR = compare_psi_movies_optical_flow(
        FlowVecSelAFWD, FlowVecSelBREV
    )
    psiMovMREV = psiMovOFMeasuresREV["Values"]
    psiMovMREV_tblock = psiMovOFMeasuresREV["Values_tblock"]

    psiMovieOFmeasures = dict(
        MeasABFWD=psiMovMFWD,
        MeasABFWD_tblock=psiMovMFWD_tblock,
        MeasABREV=psiMovMREV,
        MeasABREV_tblock=psiMovMREV_tblock,
    )
    return psiMovieOFmeasures, isBadPsiAB_blockF


def compute_edge_measure_pair_wise_all_psi(
    prd_index: int,
    neighb_prd_index: int,
    edge_num: int,
    flow_vec_prd: list[dict[str, Any]],
    flow_vec_nbr_prd: list[dict[str, Any]],
    n_edges: int,
    n_nodes: int,
    flow_vec_pct_thresh: float,
):
    num_psi = len(flow_vec_prd)

    if len(flow_vec_prd[0]["FWD"]["Vx"].shape) > 2:
        numtblocks = flow_vec_prd[0]["FWD"]["Vx"].shape[2]
    else:
        numtblocks = 1

    measure_OF_nbr_fwd = np.empty((n_edges, num_psi, num_psi))
    measure_OF_nbr_rev = np.empty((n_edges, num_psi, num_psi))

    if numtblocks > 1:
        measure_OF_nbr_fwd_tblock = np.empty((n_edges, num_psi, num_psi * numtblocks))
        measure_OF_nbr_rev_tblock = np.empty((n_edges, num_psi, num_psi * numtblocks))
    else:
        measure_OF_nbr_fwd_tblock = np.empty(0)
        measure_OF_nbr_rev_tblock = np.empty(0)

    bad_nodes_psis_block = np.zeros((n_nodes, num_psi))
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

            psi_movie_OF_measures, is_bad_psi_AB_block = (
                compute_measures_psi_movies_optical_flow(
                    flow_vec_prd_fwd,
                    flow_vec_nbr_prd_fwd,
                    flow_vec_nbr_prd_rev,
                )
            )

            measure_OF_nbr_fwd[edge_num][i_psi, psinum_neighb] = psi_movie_OF_measures[
                "MeasABFWD"
            ]
            measure_OF_nbr_rev[edge_num][i_psi, psinum_neighb] = psi_movie_OF_measures[
                "MeasABREV"
            ]

            if numtblocks > 1:
                bad_nodes_psis_block[prd_index, i_psi] = -100 * np.sum(
                    is_bad_psi_AB_block[0]
                )
                bad_nodes_psis_block[neighb_prd_index, psinum_neighb] = -100 * np.sum(
                    is_bad_psi_AB_block[1]
                )

                t = psinum_neighb * numtblocks
                print("t", t, "numtblocks", numtblocks)
                measure_OF_nbr_fwd_tblock[edge_num][i_psi, t : t + numtblocks] = (
                    np.transpose(psi_movie_OF_measures["MeasABFWD_tblock"])
                )
                measure_OF_nbr_rev_tblock[edge_num][i_psi, t : t + numtblocks] = (
                    np.transpose(psi_movie_OF_measures["MeasABREV_tblock"])
                )

    measureOFCurrNbrEdge = np.hstack(
        (measure_OF_nbr_fwd[edge_num], measure_OF_nbr_rev[edge_num])
    )

    if numtblocks > 1:
        measureOFCurrNbrEdge_tblock = np.hstack(
            (measure_OF_nbr_fwd_tblock[edge_num], measure_OF_nbr_rev_tblock[edge_num])
        )
    else:
        measureOFCurrNbrEdge_tblock = []

    return dict(
        measureOFCurrNbrEdge=measureOFCurrNbrEdge,
        measureOFCurrNbrEdge_tblock=measureOFCurrNbrEdge_tblock,
        badNodesPsisBlock=bad_nodes_psis_block,
    )


def compute_edge_measures_all(
    n_nodes: int,
    edges: list[tuple[int, int]],
    edge_nums: list[int] | npt.NDArray[np.int_],
    n_psi: int,
    flow_map: dict[int, dict[str, Any]],
    flow_vec_pct_thresh: float,
    ncpu: int = 1,
    progress_bar=NullEmitter(),
    progress_bounds=(0, 100),
):
    input_data = [
        dict(
            prd_index=edges[edge_num][0],
            neighb_prd_index=edges[edge_num][1],
            edge_num=edge_num,
            flow_vec_prd=flow_map[edges[edge_num][0]]["FlowVecPrD"],
            flow_vec_nbr_prd=flow_map[edges[edge_num][1]]["FlowVecPrD"],
            n_edges=len(edges),
            n_nodes=n_nodes,
            flow_vec_pct_thresh=flow_vec_pct_thresh,
        )
        for edge_num in edge_nums
    ]

    edge_measures_dict = dispatch_func(
        partial(dispatch_helper, func=compute_edge_measure_pair_wise_all_psi),
        input_data,
        "Computing Edge Measures",
        ncpu=ncpu,
        progress_bar=progress_bar,
        progress_bounds=progress_bounds,
    )

    edge_measures = np.empty(len(edges), dtype=object)
    edge_measures_tblock = np.empty(len(edges), dtype=object)
    bad_nodes_psis_block = np.zeros((n_nodes, n_psi))
    for e in edge_nums:
        bad_nodes_psis_block += edge_measures_dict[e]["badNodesPsisBlock"]
        edge_measures[e] = edge_measures_dict[e]["measureOFCurrNbrEdge"]
        edge_measures_tblock[e] = edge_measures_dict[e]["measureOFCurrNbrEdge_tblock"]

    # This rescaling step is to prevent underflow/overflow, should be checked if does not work
    scale_range = [5, 45]
    edge_measures = rescale_linear(edge_measures, edge_nums, scale_range)

    return edge_measures, edge_measures_tblock, bad_nodes_psis_block


def collate_bad_psi_tau(n_nodes: int, n_psi: int, flow_map: dict[int, dict[str, Any]]):
    bad_nodes_psis_tau = np.zeros((n_nodes, n_psi)).astype(int)
    nodes_psis_tau_IQR = np.zeros((n_nodes, n_psi)) + 5.0
    nodes_psis_tau_occ = np.zeros((n_nodes, n_psi))
    nodes_psis_tau_vals = [[None]] * n_nodes

    for prd, data in flow_map.items():
        if len(data["badNodesPsisTau"]):
            bad_nodes_psis_tau[prd, np.array(data["badNodesPsisTau"])] = -100
        nodes_psis_tau_IQR[prd, :] = data["NodesPsisTauIQR"]
        nodes_psis_tau_occ[prd, :] = data["NodesPsisTauOcc"]
        nodes_psis_tau_vals[prd] = data["NodesPsisTauVals"]

    return (
        bad_nodes_psis_tau,
        nodes_psis_tau_IQR,
        nodes_psis_tau_occ,
        nodes_psis_tau_vals,
    )


def calculate_energy_landscape(
    psinums: NDArray[Shape["Any"], Integer],
    senses: NDArray[Shape["Any"], Integer],
    taus: list[NDArray[Shape["Any"], Float]] | list[h5py.Dataset],
    states_per_coord: int,
    temperature: float,
):
    # Section II
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

    #################
    # compute energy:
    kB = 0.0019872041  # Boltzmann constant kcal / Mol / K
    rho = np.fmax(occupancy, 1)
    kT = kB * (temperature + 273.15)  # Kelvin
    E = -kT * np.log(rho)
    E = E - np.amin(E)  # shift so that lowest energy is zero

    return E, occupancy


def find_conformational_coords(
    prds: ProjectionDirections,
    nlsa_movies: list[list[NDArray[Shape["Any"], Float]]] | list[list[h5py.Dataset]],
    taus: list[list[NDArray[Shape["Any"], Float]]] | list[list[h5py.Dataset]],
    nlsa_mask: None | list[NDArray[Shape["Any,Any"], Float]],
    num_psi: int,
    return_all_output: bool = False,
    output_handle: None | h5py.File = None,
    flow_vec_pct_thresh: float = 0.95,
    allow_empty_clusters: bool = False,
    allow_bad_psi_tau: bool = True,
    ncpu: int = 1,
    progress_bar=NullEmitter(),
):
    G, Gsub = prds.prune_graphs()
    num_clusters = len(G["NodesConnComp"])
    anchor_list = prds.anchor_ids

    psinums, senses = validate_anchor_nodes(
        G, anchor_list, prds.trash_ids, prds.anchors, allow_empty_clusters
    )
    # FIXME: different return type
    if psinums is not None:
        if return_all_output:
            return psinums, senses, None, None, None, None, None
        return psinums, senses

    node_range, edge_num_range, cluster_no_anchor = get_cluster_nodes_edges(
        Gsub, anchor_list, num_clusters, allow_empty_clusters
    )
    G.update(ConnCompNoAnchor=cluster_no_anchor)

    flow_map = optical_flow_movie_list(
        nlsa_movies,
        taus,
        nlsa_mask,
        allow_bad_psi_tau,
        ncpu,
        progress_bar=progress_bar,
        progress_bounds=(0, 50),
    )

    edge_measures, edge_measures_tblock, bad_nodes_psis_block = (
        compute_edge_measures_all(
            G["nNodes"],
            G["Edges"],
            edge_num_range,
            num_psi,
            flow_map,
            flow_vec_pct_thresh,
            ncpu,
            progress_bar=progress_bar,
            progress_bounds=(50, 99),
        )
    )

    # This data is collected but was not used in the production code (params.use_pruned_graph was False)
    bad_nodes_psis_tau, nodes_psis_tau_IQR, nodes_psis_tau_occ, nodes_psis_tau_vals = (
        collate_bad_psi_tau(G["nNodes"], num_psi, flow_map)
    )

    options = BeliefPropagationOptions()
    node_state_bp, psinums, senses, opt_node_bel, node_belief = belief_propagation(
        prds, num_psi, G, options, edge_measures, bad_nodes_psis_block
    )

    if output_handle:
        edges = G["Edges"]
        for index in flow_map.keys():
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

            if "flow_data" in group:
                print(f"Deleting existing 'flow_data' group in 'prd_{index}'")
                del group["flow_data"]
                group.create_group("flow_data")
            else:
                group = group.create_group("flow_data")

            edge_measure_indices = np.where(
                (edges[:, 0] == index) | (edges[:, 1] == index)
            )[0]
            edge_measures_local = dict()
            for edge_index in edge_measure_indices:
                i = edges[edge_index][0]
                if i == index:
                    i = edges[edge_index][1]

                edge_measures_local[i] = edge_measures[edge_index]

            recursive_dict_to_hdf5(
                group,
                dict(
                    flow_map=flow_map[index],
                    node_belief=node_belief[:, index],
                    edge_measures=edge_measures_local,
                    bad_nodes_psis=bad_nodes_psis_block[index, :],
                ),
                overwrite=True,
            )

        output_handle.flush()

    progress_bar.emit(100)

    if return_all_output:
        return (
            psinums,
            senses,
            flow_map,
            edge_measures,
            edge_measures_tblock,
            bad_nodes_psis_block,
            node_belief,
        )
    else:
        return psinums, senses
