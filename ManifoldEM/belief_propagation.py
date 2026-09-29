import copy
from nptyping import NDArray, Shape, Integer, Float64, Object, Bool
import numpy as np
from typing import Any
from dataclasses import dataclass
from ManifoldEM.data_store import ProjectionDirections, Sense, Anchor


def node_potential(M: NDArray, beta: float):
    return np.exp(np.dot(beta, M))


def edge_potential(M: NDArray, sigma: float):
    return np.exp(-M / (2.0 * sigma**2))


def MRF_generate_potentials(
    edges: NDArray[Shape["Any,2"], Integer],
    n_nodes: int,
    anchor_nodes: list[int],
    anchor_node_measures: NDArray[Shape["Any,2"], Float64],
    edge_measures: NDArray[Shape["Any"], Object],
    num_psi: int,
    node_potential_beta: float = 1.0,
    edge_potential_sigma: float = 1.2,
):
    max_state = 2 * num_psi
    n_edges = edges.shape[0]

    # generate node potentials
    # uniform 'prior' for all unobserved nodes
    node_pots = np.zeros((max_state, n_nodes))
    for i in range(len(anchor_nodes)):
        node_pots[:, anchor_nodes[i]] = anchor_node_measures[:, i]

    # for known nodes use the provided potential
    node_pot = node_potential(node_pots, node_potential_beta)

    # generate edge potentials
    edge_pot = np.zeros((n_edges, max_state, max_state)) + 1e-10

    # flip the dimensions for Python conv
    # For the empty edges (fow which we do not have any measures) , we cannot just leave it as zeros as it will cause
    # Nan error because of 'Normalize' function. Se we add 'eps' to zeros
    for e in range(n_edges):  # change to edgeNumRange later
        if edge_measures[e] is not None:
            meas = edge_potential(edge_measures[e], edge_potential_sigma)
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

    # Anchor.CC is 1-based; state_index maps it to the 0-based BP state (reverse senses offset by num_psi)
    for index, anchor_id in enumerate(fwd_anchor_ids):
        measures_fwd[anchor[anchor_id].state_index(num_psi), index] = measure_val

    for index, anchor_id in enumerate(rev_anchor_ids):
        measures_rev[anchor[anchor_id].state_index(num_psi), index] = measure_val

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


def create_node_order(adjacency_matrix: NDArray[Shape["Any,Any"], Bool], anchor_nodes):
    """Visit the nodes breadth-first from the anchors, then the unreached ones."""
    neighbor_lists = []
    for row in range(adjacency_matrix.shape[0]):
        neighbor_lists.append(np.argwhere(adjacency_matrix[row])[:, 1].tolist())

    return get_multi_anchor_traversal_neighbor_list(anchor_nodes, neighbor_lists)


@dataclass
class BeliefPropagationOptions:
    """Sum-product belief propagation, without damping, until the message residual is below `tol`."""

    tol: float = 1e-4
    max_iter: int = 300


class MRFBeliefPropagation:
    def __init__(
        self,
        options: BeliefPropagationOptions,
        anchor_nodes: list[int],
        G: dict[str, Any],
    ):
        self.options = options
        self.anchor_nodes = copy.copy(anchor_nodes)
        self.G = copy.copy(G)
        # this is updated with each iteration and is the last message before final iteration or convergence
        self.old_message = np.empty(shape=(0, 0))
        # % this is updated with each iteration and is the final message after the final iteration or convergence
        self.new_message = np.empty(shape=(0, 0))
        # sum(abs(new_message - old_message));
        self.error = []
        self.iter = 1
        # is set to 1 if converged, otherwise 0
        self.convergence = 0
        self.convergence_iter = np.Inf

        self.initialize_message()

    def initialize_message(self):
        # all nodes have the same number of states, maxState: uniform distribution
        unif_msg = (
            np.ones((self.G["maxState"], 2 * self.G["nEdges"])) / self.G["maxState"]
        )

        self.new_message = copy.copy(unif_msg)
        self.old_message = copy.copy(unif_msg)
        self.convergence = 0

    @staticmethod
    def get_edge_idxs_from_node(G, n):
        # EdgeIdx has indices for all edges being treated as distinct (directed)
        # so edge i-j has a different id say m than edge j-i which is n, regardless
        # of the type of Adjacency matrix, undirected or directed
        # directed edge info # still has indexing from 1 to nEdges
        edges = G["EdgeIdx"][n, :].todense()
        # remove the zero values, note that EdgeIdx had indexing from 1 as matlab so 0 means no edge
        edges = edges[edges != 0]
        n_edges = G["nEdges"]

        # FIXME: -1 is because they encoded edges in 1 indexing for direct comparison to
        # matlab, this should be changed
        edge_idxs_undirected = np.array((edges - 1) % G["nEdges"]).flatten()
        edge_idxs_directed = np.array(edges).flatten() - 1
        edge_idxs_rev = np.array((edges + n_edges - 1) % (2 * n_edges)).flatten()

        return (edge_idxs_undirected, edge_idxs_directed, edge_idxs_rev)

    @staticmethod
    def normalize(M, dim: int = 1):
        if dim == 0:
            z = np.sum(M.flatten())
        else:
            z = np.sum(M, axis=dim - 1)

        return np.divide(M, z)

    def eval(self, node_pot, edge_pot):
        print("\nNow performing Belief Propagation with sum-product ...")

        n_states = self.G["nStates"]
        edges = self.G["Edges"]
        graph_node_order = self.G["graphNodeOrder"]

        # %%% Belief propagation iterations
        for self.iter in range(self.options.max_iter):
            print(f"Belief Propagation Iteration {self.iter},")
            # Each node sends a message to each of its neighbors
            # the nodes are ordered (default:sequential, 1...nNodes; min. spanning from a single anchor ; multi-anchor )

            for n in graph_node_order:
                # Find all neighbors of node n
                # we need directed edge info from G.EdgeIdx and send a message from node n/i to each of its neighbors
                [edge_idxs_undirected, edge_idxs_directed, edge_idxs_rev] = (
                    self.get_edge_idxs_from_node(self.G, n)
                )
                #  edgeIdxsDr(directed),edgeIdxsRev(reverse direction) should always be opposite

                t = 0
                for eij in edge_idxs_undirected:  # undirected;
                    eIdxRev = edge_idxs_rev[t]  # reverse direction;
                    # G.Edges has undirected edge info, edge i-j and j-i have same node ordering i<j
                    i, j = edges[eij]

                    # edge Potential for edge eij
                    e_pot_ij = edge_pot[eij, : n_states[i], : n_states[j]]

                    if n == i:
                        e_pot_ij = e_pot_ij.T

                    # Compute product of all incoming messages to node i except from node j
                    incoming_msg_prod = node_pot[: n_states[n], n]

                    # eij is undirected number so eij is always <=G.nEdges
                    kNbrOfiNOTj = edge_idxs_directed[
                        (edge_idxs_directed != eij)
                        & (edge_idxs_directed != eij + self.G["nEdges"])
                    ]
                    if len(kNbrOfiNOTj) > 0:
                        incoming_msg_prod = incoming_msg_prod * (
                            self.new_message[: n_states[n], kNbrOfiNOTj].prod(axis=1)
                        )

                    # Compute and update the new message
                    self.update_message(e_pot_ij, incoming_msg_prod, n, eij, eIdxRev)

                    t += 1

            self.check_convergence()

            if np.isnan(self.error):
                break
            if self.convergence and np.isinf(self.convergence_iter):
                self.convergence_iter = self.iter
                break

            #  update the old message to the latest message
            self.old_message = self.new_message.copy()

        if not np.isnan(self.error):
            print("Belief propagation is completed")

            if self.convergence:
                print(f"Belief propagation converged in {self.iter+1} iteration(s)")
            else:
                print(
                    f"Belief propagation did not converge after {self.iter+1} iteration(s).The beliefs can be inaccurate.\n"
                )
            print(f"Message residual at final iter {self.iter+1} = {self.error}")

            print("Computing the beliefs...")
            return self.compute_node_belief(node_pot)
        else:
            raise ValueError(
                "NaN error encountered. Check your node and edge potential values."
            )

    def update_message(self, e_pot, msg_prod, n, e_idx, e_idx_rev):
        new_message_prod = e_pot @ msg_prod

        enodes = self.G["Edges"][e_idx, :]
        # this works because we have only two elements (nodes) in an edge.
        nt = enodes[enodes != n][0]

        n_states = self.G["nStates"][nt]
        self.new_message[0:n_states, e_idx_rev] = self.normalize(new_message_prod)

    def check_convergence(self):
        self.error = np.sum(
            np.abs(self.new_message.flatten() - self.old_message.flatten())
        )

        print(f"Message residual at iter {self.iter} = {self.error}")
        if np.isnan(self.error):
            print("Message values contain NaN:Check Node/Edge Potential Values.")

        # we could do the error < tol checking in
        # the actual bp-loop and not within this function
        if self.error < self.options.tol:
            self.convergence = 1
        else:
            self.convergence = 0

    def compute_node_belief(self, nodePot):
        G = self.G
        n_nodes = G["nNodes"]
        n_states = G["nStates"]

        # % Node belief using the converged/final messages from all neighbors
        prod_of_messages = np.zeros((G["maxState"], G["nNodes"]))
        for n in range(n_nodes):
            # neighbors of node n
            _, edge_idxs_directed, _ = self.get_edge_idxs_from_node(G, n)
            prod_of_messages[: n_states[n], n] = nodePot[: n_states[n], n] * np.prod(
                self.new_message[: n_states[n], edge_idxs_directed], axis=1
            )

        return self.normalize(prod_of_messages)


def get_psi_senses_from_node_labels(node_state, num_psis):
    psinums = node_state % num_psis
    psinums[psinums == 0] = num_psis
    senses = (node_state <= num_psis) + (node_state > num_psis) * (-1)

    return (psinums, senses)


def belief_propagation(
    prds: ProjectionDirections,
    num_psi: int,
    G: dict[str, Any],
    BPoptions: BeliefPropagationOptions,
    edge_measures,
    anchor_node_pot_valexp: float = 110.0,
):
    """
    Returns each PrD's psinum and sense (-1 and 0 for the PrDs of the components listed in
    G["ConnCompNoAnchor"]), and the node beliefs, (2 * num_psi, n_nodes).
    """
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

    print("nodePot.shape:", node_pot.shape, "edgePot.shape", edge_pot.shape)

    # Global optimization with Belief propagation
    # Local pairwise measures for the projection direction/psi Topos and movies
    # are encoded in the undirected probabilistic graphical model as Markov Random Field(MRF)
    options = copy.deepcopy(BPoptions)
    # ;%.98;%0.99; %0.99 use damping factor (< 1) when message oscillates and do not converge

    G["anchorNodes"] = anchor_nodes
    G["graphNodeOrder"] = create_node_order(G["AdjMat"], anchor_nodes)

    node_belief = MRFBeliefPropagation(options, anchor_nodes, G).eval(node_pot, edge_pot)

    opt_node_labels = np.argsort(-node_belief, axis=0)
    node_state_bp = opt_node_labels[0, :]  # max-marginal

    # %%%%% Determine the Psi's and Senses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print("\nDetermining the psinum and senses from node labels ...")
    node_state_bp = node_state_bp + 1  # indexing from 1 as matlab
    psinums_bp, senses_bp = get_psi_senses_from_node_labels(node_state_bp, num_psi)

    psinums_cc = np.zeros((1, G["nNodes"]), dtype="int")
    senses_cc = np.zeros((1, G["nNodes"]), dtype="int")
    noAnchor_cc = G["ConnCompNoAnchor"]

    nodes_empty_meas = []
    for c in noAnchor_cc:
        nodes_empty_meas.append(G["NodesConnComp"][c])

    nodes_empty_meas = [y for x in nodes_empty_meas for y in x]
    nodes_empty_meas = np.array(nodes_empty_meas)
    print("nodesEmptyMeas:", nodes_empty_meas)

    psinums_cc[:] = psinums_bp - 1  # python starts with 0
    senses_cc[:] = senses_bp

    psinums_cc = psinums_cc.flatten()
    senses_cc = senses_cc.flatten()

    # if no measurements for a node,as it was an isolated node
    if len(nodes_empty_meas) > 0:
        # put psinum/senses value to -1, for the nodes 'nodesEmpty' for which there were no calculations done.
        psinums_cc[nodes_empty_meas] = -1
        senses_cc[nodes_empty_meas] = 0

    print("Total bad psinum PDs marked:", np.sum(psinums_cc == -1))
    print("psinums_cc", psinums_cc)
    print("senses_cc", senses_cc)

    return psinums_cc, senses_cc, node_belief
