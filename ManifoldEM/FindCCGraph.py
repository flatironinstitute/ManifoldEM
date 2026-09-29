import numpy as np
from typing import Set, Any, Dict, Tuple, Union, List
from nptyping import NDArray, Shape

from ManifoldEM.params import params

from scipy.sparse import csr_matrix, tril, lil_matrix
from scipy.sparse.csgraph import connected_components

'''
Copyright (c) Columbia University Suvrajit Maji 2019
'''
'''
function
G = CreateGraphStruct(Xp, AdjMat, nStates, pwDist, numNbr, epsilon)
% Create graph structure
% Input:
% Xp: 3xn starting points(coordinates) on S2, n is the number of nodes
% nStates: number of states each node can have
% Note:: we assume that all nodes have same number of states here.
% numNbr: number of nearest neighbor we would like to have for each
% neighbor
% Output:
% G: graph structure
%
%
% Suvrajit Maji, sm4073 @ cumc.columbia.edu
% Columbia University
% Created: Feb 02, 2018. Modified:Jan 25, 2019
Python version Hstau Liao copyright 2018
'''

# FIXME: just... fix it.
def CreateGraphStruct(nStates: int,
                      pwDist: Union[List[float], NDArray[Shape["*"], np.float64]],
                      epsilon: Union[float, None],
                      *argv: Any) -> Dict[str, Any]:
    if type(pwDist) is list:
        pwDist = np.array(pwDist)

    if argv:
        AdjMat = argv[0]
    else:
        AdjMat = np.empty(0)

    if pwDist.shape[0] > 0:
        nNodes = pwDist.shape[0]
    else:
        nNodes = AdjMat.shape[0]

    Nodes = range(nNodes)
    G = dict(nNodes=nNodes, Nodes=Nodes)

    # create state for each node
    if np.isscalar(nStates):
        nStates = nStates * np.ones(nNodes, dtype='int')

    G.update(nStates=nStates)
    maxState = np.max(nStates)
    G.update(maxState=maxState)

    nnMat = np.empty((G['nNodes'], ), dtype=object)
    if not argv:  # adj matrix absent
        # create the connections from neighbor search
        Adj = (pwDist <= epsilon) * (pwDist != 0)
        Adj = csr_matrix(Adj)
        # form the graph model
        for n in range(nNodes):

            nnMat[n] = np.nonzero(Adj[n, :])[1]

        # if it is not symmetric
        Adj = Adj + Adj.T
        AdjMat = (Adj > 0)
    else:
        for n in range(nNodes):
            nnMat[n] = np.nonzero(AdjMat[n, :])[1]

    AdjMat = csr_matrix(AdjMat)

    # 1.
    # create edge indices
    ni, nj = np.nonzero(AdjMat)

    Edges = np.vstack((nj, ni)).T

    I = np.lexsort((Edges[:, 1], Edges[:, 0]))
    Edges = Edges[I, :]

    Edges = Edges[np.nonzero(Edges[:, 0] < Edges[:, 1])]
    print('Number of Graph Edges:', Edges.shape)

    # 2.
    ni, nj = np.nonzero(tril(AdjMat))

    ### to make the output same as matlab implementation
    nij = np.c_[ni, nj]
    nids = np.lexsort((nij[:, 0], nij[:, 1]))
    nij_s = nij[nids, :]
    ni = nij_s[:, 0]
    nj = nij_s[:, 1]

    nEdges = len(ni)
    val_e = np.arange(
        nEdges) + 1  # for now, to compare with matlab we need EdgeIdx to contain 1 to nEdges even for python

    Ni = np.hstack((ni, nj)).T
    Nj = np.hstack((nj, ni)).T
    Val = np.hstack((val_e, val_e + nEdges)).T
    EdgeIdx = csr_matrix((Val, (Ni, Nj)), shape=AdjMat.shape)

    G.update(Nodes=Nodes,
             nNodes=nNodes,
             epsilon=epsilon,
             nnMat=nnMat,
             AdjMat=AdjMat,
             Edges=Edges,
             nEdges=nEdges,
             EdgeIdx=EdgeIdx,
             nStates=nStates)
    return G


'''
function [Gsub,G] = getSubGraph(G,nodes)
% Obtain subgraph of a graph G given the subset of nodes to be included
% Input:
%   G: Input graph / adjacency
%   nodes: subset of nodes to be included into the subgraph
%
% Output:
%   Gsub: subgraph, returns cell array of graph structure(s) for multiple
%   subgraphs
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Mar 12,2018. Modified: Jan 25,2019
%
%
'''


def getSubGraph(G) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("\nPerforming connected component analysis.")
    A = G['AdjMat']
    S, C = connected_components(A)
    G.update(NodesConnComp=[])
    for i in np.arange(S):
        G['NodesConnComp'].append(np.nonzero(C == i)[0])

    numConnComp = len(G['NodesConnComp'])
    print("Number of connected components:", numConnComp)
    if numConnComp > 5:  # typically there would be less than 5 connected components for a good coverage dataset
        print('Warning:Too many connected components.')
        print('The dataset may have regions of sparse S2 coverage or, check the input tesselation parameters.')

    Gsub = []
    for i in range(numConnComp):
        if len(G['NodesConnComp'][i]) == 1:
            if numConnComp <= 5:  #only provide this print statement if number of components are 5 or less
                print('Singlet Node in connected component', i)

        nodes = G['NodesConnComp'][i]
        Asub = G['AdjMat'][nodes][:, nodes]
        Gsub.append(CreateGraphStruct(G['maxState'], [], G['epsilon'], Asub))

        Gsub[i]['originalNodes'] = nodes
        einds = np.in1d(G['Edges'][:, 0], nodes) | np.in1d(G['Edges'][:, 1], nodes)
        Gsub[i]['originalEdgeList'] = np.nonzero(einds)

    return (Gsub, G)


def CalcPairwiseDistS2(X):
    '''
    Pairwise Euclidean distances between the columns of a 3xN coordinate matrix.

    Author Suvrajit Maji, sm4073@cumc.columbia.edu
    Columbia University
    Created: Dec 2017.
    Modified: Feb02, 2018

    Python version Hstau Liao copyright 2018
    '''
    # Separate copies keep numpy on the general matrix product: for operands that share memory
    # it switches to a symmetric BLAS routine that rounds differently.
    U = X.copy()
    V = X.copy()

    # pairwise Euclidean distance
    Dsq = np.sum(U * U, axis=0).T + np.sum(V * V, axis=0) - 2 * np.dot(U.T, V)
    # clip rounding noise so that identical points get exactly zero distance
    Dsq[Dsq < 1e-6] = 0.0
    pwDist = np.sqrt(Dsq)
    return pwDist


def prune(G: Dict[str, Any], trash_ids: Set[int], num_psis: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_nodes_tot = G['nNodes']
    n_nodes_left = n_nodes_tot - len(trash_ids)
    print(f"Number of isolated nodes in the graph after pruning: {len(trash_ids)}")
    trash_list = list(trash_ids)

    max_state = 2 * num_psis
    if n_nodes_left > 1:
        epsilon = G['epsilon']  # save it later after update

        print('Number of Graph Edges before prunning:', G['nEdges'])
        # prune edges corresponding to the bad nodes with actually removing those bad nodes by disconnecting the edges
        # in and out of those specified nodes
        new_adj_mat = lil_matrix(G['AdjMat'])
        new_adj_mat[trash_list, :] = 0
        new_adj_mat[:, trash_list] = 0

        new_adj_mat = csr_matrix(new_adj_mat)

        # Updated graph info
        G = CreateGraphStruct(G['maxState'], [], None, new_adj_mat)  # june 2020

        # re-insert the epsilon
        G['epsilon'] = epsilon

        print('Number of Graph Edges after pruning:', G['nEdges'])
        # If the min distance /epsilon values are different during initial and pruned graph creation
        # the actual number of edges may or may not decrease after pruning, the nodes that were pruned will become
        # isolated
    else:
        print('Single PrD. Empty graph structure created with one node.')
        G = CreateGraphStruct(max_state, [0], 0)

    # Determine if there are multiple connected components / subgraph
    # proceed to the pairwise measurements only after we are fine with the connected components
    Gsub, G = getSubGraph(G)

    return G, Gsub


def op(CG, nG, S20_th):
    numPDs = len(CG)
    print("Number of PDs:", numPDs)

    #Number of Projection directions
    PrDs = range(numPDs)

    maxState = 2 * params.num_psi  # Up and Down

    if numPDs > 1:
        # Setting up the graph structure
        pwDist = CalcPairwiseDistS2(S20_th[:, PrDs])

        mindist = np.min(pwDist[np.nonzero(pwDist)])  # 2*shAngWidth

        epsilonBall = mindist * nG / numPDs
        epsilon = min(max(mindist + params.eps, epsilonBall),
                      mindist * 2 * np.sqrt(2) + params.eps)  #mindist + eps <= epsilon <= mindist *2*sqrt(2) + eps
        print("Neighborhood epsilon:", epsilon)

        # Updated graph info
        G = CreateGraphStruct(maxState, pwDist, epsilon)
    else:
        G = CreateGraphStruct(maxState, [0], 0)

    # Determine if there are multiple connected components / subgraph
    # proceed to the pairwise measurements only after we are fine with the connected components
    Gsub, G = getSubGraph(G)

    return G, Gsub
