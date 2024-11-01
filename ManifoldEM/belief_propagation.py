import copy
from nptyping import NDArray, Shape, Integer, Float64, Object, Bool
import numpy as np
from typing import Any

from ManifoldEM.data_store import ProjectionDirections, Sense, Anchor


def node_potential(M: NDArray, beta: float):
    return np.exp(np.dot(beta, M))


def transform(M: NDArray):
    sigma = 1.2
    return np.exp(-M / (2.0 * sigma**2))


def MRF_generate_potentials(
    edges: NDArray[Shape["Any,2"], Integer],
    n_nodes: int,
    anchor_nodes: list[int],
    anchor_node_measures: NDArray[Shape["Any,2"], Float64],
    edge_measures: NDArray[Shape["Any"], Object],
    num_psi: int,
):
    max_state = 2 * num_psi
    n_edges = edges.shape[0]

    # generate node potentials
    # uniform 'prior' for all unobserved nodes
    node_pots = np.zeros((max_state, n_nodes))
    for i in range(len(anchor_nodes)):
        node_pots[:, anchor_nodes[i]] = anchor_node_measures[:, i]

    # for known nodes use the provided potential
    node_pot = node_potential(node_pots, 1)

    # generate edge potentials
    edge_pot = np.zeros((n_edges, max_state, max_state)) + 1e-10

    # flip the dimensions for Python conv
    # For the empty edges (fow which we do not have any measures) , we cannot just leave it as zeros as it will cause
    # Nan error because of 'Normalize' function. Se we add 'eps' to zeros
    for e in range(n_edges):  # change to edgeNumRange later
        if edge_measures[e] is not None:
            meas = transform(edge_measures[e])
            edge_pot[e, :, :] = np.vstack(
                (meas, np.hstack((meas[:, num_psi:], meas[:, :num_psi])))
            )

    return (node_pot, edge_pot)


def initialize_anchor_node_measures(
    anchor: dict[int, Anchor], num_psi: int, measure_val: float
) -> tuple[list[int], NDArray[Shape["Any,Any"], Float64]]:
    """
    Initializes the anchor node measures for the Markov Random Field (MRF)

    Parameters
    ----------
    anchor: dict[int, Anchor]
    num_psi: int

    Returns
    -------
    tuple
        list[int]
            [forward_sensed_anchor_ids] + [reverse_sensed_anchor_ids]
        ndarray
            (2*num_psi) x n_anchors array of measures
    """
    max_state = 2 * num_psi
    fwd_anchor_ids = list(
        dict(filter(lambda keyval: keyval[1].sense == Sense.FWD, anchor.items())).keys()
    )
    rev_anchor_ids = list(
        dict(filter(lambda keyval: keyval[1].sense == Sense.REV, anchor.items())).keys()
    )
    anchor_nodes = fwd_anchor_ids + rev_anchor_ids

    measures_fwd = np.zeros((max_state, len(fwd_anchor_ids)))
    measures_rev = np.zeros((max_state, len(rev_anchor_ids)))

    print(f"anchorNodes: {anchor_nodes}")

    # FIXME: I assume the CC field should be indexed from 0 (i.e. CC-1)
    for index, anchor_id in enumerate(fwd_anchor_ids):
        measures_fwd[anchor[anchor_id].CC, index] = measure_val

    for index, anchor_id in enumerate(rev_anchor_ids):
        measures_rev[anchor[anchor_id].CC + num_psi, index] = measure_val

    return anchor_nodes, np.hstack((measures_fwd, measures_rev))


def get_multi_anchor_traversal_neighbor_list(
    anchors: list[int], neighbor_lists: list[list[int]]
):
    ordered_nodes = copy.copy(anchors)
    cur_nodes = copy.copy(anchors)
    next_nodes = []
    n_nodes = len(neighbor_lists)

    for _ in range(n_nodes):
        for node_i in cur_nodes:
            j_neighb = neighbor_lists[node_i]

            for probe in j_neighb:
                if probe not in ordered_nodes:
                    ordered_nodes.append(probe)
                    next_nodes.append(probe)

        cur_nodes = next_nodes
        next_nodes = []

    # add the remaining nodes which were not visited to the final list
    ordered_nodes = ordered_nodes + list(set(range(n_nodes)) - set(ordered_nodes))

    return np.array(ordered_nodes)


def create_node_order(
    adjacency_matrix: NDArray[Shape["Any,Any"], Bool],
    anchor_nodes,
    node_order_type: str = "default",
):
    if node_order_type == "default":
        node_order = np.array(range(adjacency_matrix.shape[0]))
    elif node_order_type == "multiAnchor":
        neighbor_lists = []
        for row in range(adjacency_matrix.shape[0]):
            neighbor_lists.append(np.argwhere(adjacency_matrix[row])[:, 1].tolist())

        node_order = get_multi_anchor_traversal_neighbor_list(
            anchor_nodes, neighbor_lists
        )
    else:
        msg = f"Invalid node_order_type '{node_order_type}'. Expected 'default' or 'multiAnchor'"
        raise ValueError(msg)

    return node_order


def augment_potential_bad_nodes(
    bad_nodes_psis: NDArray[Shape["Any,Any"], Integer],
    node_pot: NDArray[Shape["Any,Any"], Float64],
    num_psi: int,
    bad_node_pot_val: float,
):
    nodes_all_bad_psis = []
    for n in range(bad_nodes_psis.shape[0]):
        # row has prd numbers, column has psi number so shape is (num_prds,2)
        # remember that badNodePsis has index starting with 1 ??
        bad_psis = np.nonzero(bad_nodes_psis[n, :] <= -100)[0]

        for k in bad_psis:
            if k < num_psi:
                node_pot[k, n] = bad_node_pot_val
                node_pot[k + num_psi, n] = bad_node_pot_val
            else:
                node_pot[k, n] = bad_node_pot_val
                node_pot[k - num_psi, n] = bad_node_pot_val

        if len(bad_psis) == bad_nodes_psis.shape[1]:
            # all columns should be bad
            # if len(badPsis)>=badNodesPsis2.shape[1]-1: # all columns, or less one bad ?
            nodes_all_bad_psis.append(n)

    return np.array(nodes_all_bad_psis)


def belief_propagation(
    prds: ProjectionDirections,
    num_psi: int,
    G: dict[str, Any],
    BPoptions: dict[str, Any],
    edge_measures,
    bad_nodes_psis,
    cc,
    enforce_bad_state_removal: bool = False,
    anchor_node_pot_valexp: float = 110.0,
    bad_node_pot_val: float = 1.0e-20,
):
    # Generate the node and edge potentials for the Markov Random Field
    # always make sure that the number of states maxState = 2*nPsiModes, because
    # we have two levels: up and down state for each psiMode.
    max_state = 2 * num_psi

    # update the nPsiModes in case p.num_psis is changed in the later steps
    G.update(nPsiModes=num_psi)
    G.update(maxState=max_state)

    # Sort anchor nodes and initialize the anchor node measures
    anchor_nodes, anchor_node_measures = initialize_anchor_node_measures(
        prds.anchors, num_psi, anchor_node_pot_valexp
    )

    node_pot, edge_pot = MRF_generate_potentials(
        G["Edges"],
        G["nNodes"],
        anchor_nodes,
        anchor_node_measures,
        edge_measures,
        num_psi,
    )

    nodes_all_bad_psis = augment_potential_bad_nodes(
        bad_nodes_psis, node_pot, num_psi, bad_node_pot_val
    )
    print("nodesAllBadPsis", len(nodes_all_bad_psis))
    print("nodePot.shape:", node_pot.shape, "edgePot.shape", edge_pot.shape)

    # Global optimization with Belief propagation
    # Local pairwise measures for the projection direction/psi Topos and movies
    # are encoded in the undirected probabilistic graphical model as Markov Random Field(MRF)
    options = copy.deepcopy(BPoptions)
    # ;%.98;%0.99; %0.99 use damping factor (< 1) when message oscillates and do not converge

    G["anchorNodes"] = anchor_nodes
    G["graphNodeOrder"] = create_node_order(G["AdjMat"], anchor_nodes, "multiAnchor")

    BPalg = createBPalg(G, options)
    BPalg["anchorNodes"] = anchor_nodes

    nodeBelief, edgeBelief, BPalg = MRFBeliefPropagation.op(BPalg, node_pot, edge_pot)

    nodeBeliefR = nodeBelief

    if enforce_bad_state_removal:
        nodeBeliefR = nodeBelief
        badS = bad_nodes_psis == -100
        print(badS[0, :], np.shape(badS))
        badStates = np.hstack((badS, badS)).T  # FWD + REV states

        nodeBeliefR[badStates] = 0.0

    OptNodeLabels = np.argsort(-nodeBeliefR, axis=0)
    nodeStateBP = OptNodeLabels[0, :]  # %max-marginal
    OptNodeBel = nodeBeliefR[nodeStateBP, range(0, len(nodeStateBP))]

    # %%%%% Determine the Psi's and Senses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print("\nDetermining the psinum and senses from node labels ...")
    nodeStateBP = nodeStateBP + 1  # indexing from 1 as matlab

    psinumsBP, sensesBP = getPsiSensesfromNodeLabels(nodeStateBP, num_psi)

    psinums_cc = np.zeros((1, G["nNodes"]), dtype="int")
    senses_cc = np.zeros((1, G["nNodes"]), dtype="int")

    noAnchorCC = G["ConnCompNoAnchor"]

    nodesEmptyMeas = []
    for c in noAnchorCC:
        nodesEmptyMeas.append(G["NodesConnComp"][c])

    nodesEmptyMeas = [y for x in nodesEmptyMeas for y in x]
    nodesEmptyMeas = np.array(nodesEmptyMeas)
    print("nodesEmptyMeas:", nodesEmptyMeas)

    psinums_cc[:] = psinumsBP - 1  # python starts with 0
    senses_cc[:] = sensesBP

    psinums_cc = psinums_cc.flatten()
    senses_cc = senses_cc.flatten()

    # if no measurements for a node,as it was an isolated node
    if len(nodesEmptyMeas) > 0:
        # put psinum/senses value to -1, for the nodes 'nodesEmpty' for which there were no calculations done.
        psinums_cc[nodesEmptyMeas] = -1
        senses_cc[nodesEmptyMeas] = 0

    # if all psi-states for a node was bad
    if len(nodes_all_bad_psis) > 0:
        psinums_cc[nodes_all_bad_psis] = -1
        senses_cc[nodes_all_bad_psis] = 0

    print("Total bad psinum PDs marked:", np.sum(psinums_cc == -1))

    print("psinums_cc", psinums_cc)
    print("senses_cc", senses_cc)

    return (nodeStateBP, psinums_cc, senses_cc, OptNodeBel, nodeBelief)
