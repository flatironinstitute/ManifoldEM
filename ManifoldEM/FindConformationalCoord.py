import os

import numpy as np
from copy import deepcopy

from ManifoldEM import myio
from ManifoldEM.data_store import data_store
from ManifoldEM.FindCCGraph import prune
from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.CC import ComputePsiMovieEdgeMeasurements, runGlobalOptimization

''' Suvrajit Maji,sm4073@cumc.columbia.edu
    Columbia University
    Created: Dec 2017. Modified:Aug 16,2019
'''

def force_remove(*paths):
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def op(*argv):
    params.load()

    nodeOutputFile = os.path.join(params.CC_dir, 'comp_psi_sense_nodes.txt')
    nodeBelFile1 = os.path.join(params.CC_dir, 'nodeAllStateBel_rc1.txt')
    nodeBelFile2 = os.path.join(params.CC_dir, 'nodeAllStateBel_rc2.txt')
    force_remove(params.CC_file, nodeBelFile1, nodeBelFile2)

    # if trash PDs were created manually
    prds = data_store.get_prds()
    num_trash_nodes = len(prds.trash_ids)

    # FIXME: This should be offloaded to the data_store (pref a graph class)
    # Prune graph structure
    if num_trash_nodes:
        print('Number of trash PDs', num_trash_nodes)
        prds.neighbor_graph_pruned, prds.neighbor_subgraph_pruned = \
            prune(deepcopy(prds.neighbor_graph), prds.trash_ids, params.num_psi)
    else:
        prds.neighbor_graph_pruned, prds.neighbor_subgraph_pruned = \
            deepcopy(prds.neighbor_graph), deepcopy(prds.neighbor_subgraph)

    G, Gsub = prds.neighbor_graph_pruned, prds.neighbor_subgraph_pruned

    numConnComp = len(G['NodesConnComp'])

    anchorlist = prds.anchor_ids
    print(f'Number of anchor nodes: {len(anchorlist)}')
    print(f'Anchor list: {anchorlist}')
    if set(anchorlist).intersection(set(G['Nodes'])):
        if len(anchorlist) + num_trash_nodes == G['nNodes']:
            print('\nAll nodes have been manually selected (as anchor nodes). '
                  'Conformational-coordinate propagation is not required. Exiting this program.\n')

            psinums = np.zeros((2, G['nNodes']), dtype='int')
            senses = np.zeros((2, G['nNodes']), dtype='int')

            for id, anchor in prds.anchors.items():
                psinums[0, id] = anchor.CC - 1
                senses[0, id] = anchor.sense

            for trash_index in prds.trash_ids:
                psinums[0, trash_index] = -1
                senses[0, trash_index] = 0

            print('\nFind CC: Writing the output to disk...\n')
            myio.fout1(params.CC_file, psinums=psinums, senses=senses)

            return
    else:
        print('Some(or all) of the anchor nodes are NOT in the Graph node list.')
        return

    nodelCsel = []
    edgelCsel = []
    # this list keeps track of the connected component (single nodes included) for which no anchor was provided
    connCompNoAnchor = []
    for i in range(numConnComp):
        nodesGsubi = Gsub[i]['originalNodes']
        edgelistGsubi = Gsub[i]['originalEdgeList']

        if any(x in anchorlist for x in nodesGsubi):
            nodelCsel.append(nodesGsubi.tolist())
            edgelCsel.append(edgelistGsubi[0])
        else:
            connCompNoAnchor.append(i)
            print('Anchor node(s) in connected component', i, ' NOT selected.')
            print('\nIf you proceed without atleast one anchor node for the connected component', i,
                  ', all the corresponding nodes will not be assigned with reaction coordinate labels.\n')

    G.update(ConnCompNoAnchor=connCompNoAnchor)

    nodeRange = np.sort([y for x in nodelCsel for y in x])
    edgeNumRange = np.sort([y for x in edgelCsel for y in x])

    # compute all pairwise edge measurements
    # Step 1: compute the optical flow vectors for all prds
    # Step 2: compute the pairwise edge measurements for all psi - psi movies
    # Step 3: Extract the pairwise edge measurements to be used for node-potential and edge-potential calculations
    edgeMeasures, edgeMeasures_tblock, badNodesPsisBlock = \
        ComputePsiMovieEdgeMeasurements.op(G, nodeRange, edgeNumRange, *argv)

    # Setup and run the Optimization: Belief propagation
    print('\n4.Running Global optimization to estimate state probability of all nodes ...')
    BPoptions = dict(maxProduct=0, verbose=0, tol=1e-4, maxIter=300, eqnStates=1.0, alphaDamp=1.0)

    # reaction coordinate number rc = 1,2
    psinums = np.zeros((2, G['nNodes']), dtype='int')
    senses = np.zeros((2, G['nNodes']), dtype='int')
    cc = 1
    print('\nFinding CC for Dim:1')
    nodeStateBP_cc1, psinums_cc1, senses_cc1, OptNodeBel_cc1, nodeBelief_cc1 = runGlobalOptimization.op(
        G, BPoptions, edgeMeasures, edgeMeasures_tblock, badNodesPsisBlock, cc)
    psinums[0, :] = psinums_cc1
    senses[0, :] = senses_cc1

    if params.n_reaction_coords == 2:
        cc = 2
        print('\nFinding CC for Dim:2')
        nodeStateBP_cc2, psinums_cc2, senses_cc2, OptNodeBel_cc2, nodeBelief_cc2 = runGlobalOptimization.op(
            G, BPoptions, edgeMeasures, edgeMeasures_tblock, badNodesPsisBlock, cc, nodeStateBP_cc1)
        psinums[1, :] = psinums_cc2
        senses[1, :] = senses_cc2

    # save
    print('\nFind CC: Writing the output to disk...\n')
    myio.fout1(params.CC_file, psinums=psinums, senses=senses)

    if params.n_reaction_coords == 1:  # 1 dimension
        node_list = np.empty((G['nNodes'], 4))
        node_list[:, 0] = np.arange(1, G['nNodes'] + 1)
        node_list[:, 1] = psinums[0, :] + 1
        node_list[:, 2] = senses[0, :]
        node_list[:, 3] = OptNodeBel_cc1

        # save the found psinum , senses also as text file
        # node_list is variable name with columns: if dim =1 : (PrD, CC1, S1) + (CC2, S2) if dim =2
        np.savetxt(nodeOutputFile, node_list, fmt='%i\t%i\t%i\t%f', delimiter='\t')

        nodeBels1 = np.empty((nodeBelief_cc1.T.shape[0], nodeBelief_cc1.T.shape[1] + 1))
        nodeBels1[:, 0] = range(1, G['nNodes'] + 1)
        nodeBels1[:, 1:] = nodeBelief_cc1.T
        np.savetxt(nodeBelFile1, nodeBels1, fmt='%f', delimiter='\t')

    elif params.n_reaction_coords == 2:  # 2 dimension
        node_list = np.empty((G['nNodes'], 7))
        node_list[:, 0] = np.arange(1, G['nNodes'] + 1)
        node_list[:, 1:3] = (psinums + 1).T
        node_list[:, 3:5] = senses.T
        node_list[:, 5] = OptNodeBel_cc1
        node_list[:, 6] = OptNodeBel_cc2

        # save the found psinum , senses also as text file
        # node_list is variable name with columns: if dim =1 : (PrD, CC1, S1) + (CC2, S2) if dim =2
        np.savetxt(nodeOutputFile, node_list, fmt='%i\t%i\t%i\t%i\t%i\t%f\t%f', delimiter='\t')

        nodeBels2 = np.empty((nodeBelief_cc2.T.shape[0], nodeBelief_cc2.T.shape[1] + 1))
        nodeBels2[:, 0] = range(1, G['nNodes'] + 1)
        nodeBels2[:, 1:] = nodeBelief_cc2.T
        np.savetxt(nodeBelFile2, nodeBels2, fmt='%f', delimiter='\t')

    params.project_level = ProjectLevel.FIND_CCS
    params.save()

    if argv:
        progress5 = argv[0]
        progress5.emit(int(100))
