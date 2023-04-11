import pickle

import numpy as np

from ManifoldEM import myio, p
from ManifoldEM.CC import MRFGeneratePotentials, MRFBeliefPropagation
from ManifoldEM.CC.MRFBeliefPropagation import createBPalg

'''
function list = rearrange(seeds,nn)
% function to return a list of nodes ordered according to the
% degree of separation from the nodes in the seeds array
%
% INPUT: nn is an Nx(K+1) matrix, where N is # nodes and K # neighbors
%        seeds is vector with the seed node indeces
%
% OUTPUT: list is the list of nodes ordered according to the degree of
% separation from the seed nodes.
%
% E.g., seeds = (6, 15)                        1  2  3  4
%        nn = (1, 2, 5                         5  6  7  8
%              2, 1, 3, 6                      9 10 11 12
%              3, 2,4, 7                      13 14 15 16
%              4, 3, 8
%              ...)
%
%        then list = (6,15,2,5,7,10,11,14,16,1,3,9,8,7,12,15,13,4)
% initialize list
%
% Hstau Liao
% Columbia University
% Created: Feb 24,2018. Modified:Feb 24,2018
%
'''


def rearrange(seeds, nn):
    nodelist = []
    for i in range(len(seeds)):
        nodelist.append(seeds[i])

    cur_nodes = seeds
    #seeds
    next_nodes = []  # % one degree of sepration from cur_nodes

    #%while length(list) < size(nn,1)
    for szlist in range(nn.shape[0]):
        for j in range(len(cur_nodes)):
            #%
            j_neigh = nn[cur_nodes[j], :]
            j_neigh = j_neigh[
                j_neigh !=
                -100]  #% remove -100 as neighbors, indexing starts with 0, so a negative number was used in nn for fillers.

            for k in range(len(j_neigh)):
                probe = j_neigh[k]
                if probe not in nodelist:  #%&& probe > 0
                    nodelist.append(probe)
                    next_nodes.append(probe)

        cur_nodes = next_nodes
        next_nodes = []

    #% add the remaining nodes which were not visited, to the final list
    remnodes = set(range(nn.shape[0])) - set(nodelist)

    if len(remnodes) > 0:
        nodelist = nodelist + list(remnodes)


    nodelist = np.array(nodelist)

    return nodelist


'''
function [nodeOrder,G] = createNodeOrder(G,anchorNodes,nodeOrderType)
% Generate a ordering of node numbers for visiting the nodes
% Input:
%   G: Input graph / adjacency
%   anchorNodes: The anchor nodes
%   nodeOrderType: Node order type
%       'default' :Sequential order 1...nNodes
%       'minSpan' :Minimum spanning tree order with 'BFS' or 'DFS' search
%       'multiAnchor': The nodes arrangement will be start from the anchors
%        and then progressively with the neihbors of the anchor nodes
% Output:
%   nodeOrder: Reordered node numbers
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 28,2018. Modified: Mar 01,2018
%
%
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version an debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)

'''


def createNodeOrder(G, anchorNodes, nodeOrderType):

    print('nodeOrderType:', nodeOrderType)

    if not nodeOrderType or nodeOrderType == []:
        #% default sequential order
        nodeOrderType = 'default'

    A = G['AdjMat']
    nodeOrder = np.empty(G['nNodes'])

    if nodeOrderType == 'default':
        nodeOrder = np.array(range(A.shape[0]))

    elif nodeOrderType == 'multiAnchor':
        nnMatCell = G['nnMat']
        nnMatCell = np.reshape(nnMatCell, (G['nNodes'], -1))


        #Sz = cell2mat(cellfun(@(x) size(x,2),nnMatCell,'UniformOutput',False))
        Sz = np.apply_along_axis(lambda x: len(x[0]), 1, nnMatCell)
        maxSz = max(Sz)

        #nnMat = cell2mat(cellfun(@(x) [x zeros(1,maxSz - size(x,2))],nnMatCell,'UniformOutput',false));
        nnMat = np.apply_along_axis(lambda x: np.append(x[0], -100 * (np.ones((1, maxSz - len(x[0]))))).tolist(), 1,
                                    nnMatCell).astype(int)  # put -100 as filler since indexing starts with 0

        nodeOrder = rearrange(anchorNodes, nnMat)

    return nodeOrder


'''
function [psinums,senses] = getPsiSensesfromNodeLabels(nodeState,NumPsis)
% variables for final psi-nums and senses
%psinums = zeros(2,length(PrDs));
%senses = zeros(2,length(PrDs));
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''


def getPsiSensesfromNodeLabels(nodeState, NumPsis):

    nodePsiLabels = nodeState
    psinums = (nodePsiLabels % NumPsis)
    psinums[psinums == 0] = NumPsis
    senses = (nodePsiLabels <= NumPsis) + (nodePsiLabels > NumPsis) * (-1)

    return (psinums, senses)


def readBadNodesPsisTau(badNodesPsisTaufile):
    try:
        dataR = myio.fin1(badNodesPsisTaufile)
        badNodesPsisTau = dataR['badNodesPsisTau_of']
        # make sure 'badNodesPsisTau_of' (and not 'badNodesPsisTau')  is used
        # badNodesPsisTau has only the initial estimates tau-cutoff
        # 'badNodesPsisTau_of' has updated info of bad taus after optical flow movie step ('_of') and so it is used finally
    except:
        badNodesPsisTau = np.empty((0, 0))

    return badNodesPsisTau


'''
function [nodeStateBP,psinums_cc,senses_cc,OptNodeBel] = runGlobalOptimization (G,BPoptions,edgeMeasures,cc)
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging, and extensive modifications)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''


def op(G, BPoptions, edgeMeasures, edgeMeasures_tblock, badNodesPsis, cc, *argv):

    # some settings
    anchorNodePotValexp = 110.0

    badNodePotVal = 1.0e-20
    lowNodePotVal = 1.0e-20

    enforceBadStateRemoval = 0
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Generate the node and edge potentials for the Markov Random Field
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    NumPsis = p.num_psis
    #always make sure that the number of states maxState = 2*nPsiModes, because
    #we have two levels: up and down state for each psiMode.
    maxState = 2 * NumPsis
    G.update(nPsiModes=p.num_psis)  # update the nPsiModes in case p.num_psis is changed in the later steps
    G.update(maxState=maxState)

    if cc == 1:

        # format: PrD,CC1,S1 for 1D
        # p.anch_list = np.array([[1,1,1],[2,1,-1]])  #TEMP should PrD and CC1 start from 1 or 0?
        p.anch_list = np.array(p.anch_list)



        IndStatePlusOne = p.anch_list[:, 2] == 1
        IndStateMinusOne = p.anch_list[:, 2] == -1
        anchorNodesPlusOne = p.anch_list[IndStatePlusOne, 0] - 1
        anchorNodesMinusOne = p.anch_list[IndStateMinusOne, 0] - 1
        anchorStatePlusOne = p.anch_list[IndStatePlusOne, 1] - 1
        anchorStateMinusOne = p.anch_list[IndStateMinusOne, 1] + NumPsis - 1

    elif cc == 2:
        if argv:
            nodeStateBP_cc1 = argv[0]

        # format: PrD,CC1,S1,CC2,S2 for 2D
        IndStatePlusOne = p.anch_list[:, 4] == 1
        IndStateMinusOne = p.anch_list[:, 4] == -1
        anchorNodesPlusOne = p.anch_list[IndStatePlusOne, 0] - 1
        anchorNodesMinusOne = p.anch_list[IndStateMinusOne, 0] - 1
        anchorStatePlusOne = p.anch_list[IndStatePlusOne, 3] - 1
        anchorStateMinusOne = p.anch_list[IndStateMinusOne, 3] + NumPsis - 1

    #anchorNodes = [anchorNodesPlusOne, anchorNodesMinusOne]

    anchorNodeMeasuresPlusOne = np.zeros((maxState, len(anchorNodesPlusOne)))
    anchorNodeMeasuresMinusOne = np.zeros((maxState, len(anchorNodesMinusOne)))

    anchorNodes = anchorNodesPlusOne.tolist() + anchorNodesMinusOne.tolist()
    print('anchorNodes:', anchorNodes)

    for u in range(len(anchorNodesPlusOne)):
        anchorNodeMeasuresPlusOne[anchorStatePlusOne[u], u] = anchorNodePotValexp

    for v in range(len(anchorNodesMinusOne)):
        anchorNodeMeasuresMinusOne[anchorStateMinusOne[v], v] = anchorNodePotValexp

    anchorNodeMeasures = np.hstack((anchorNodeMeasuresPlusOne, anchorNodeMeasuresMinusOne))
    nodePot, edgePot = MRFGeneratePotentials.op(G, anchorNodes, anchorNodeMeasures, edgeMeasures, edgeMeasures_tblock)


    # Set potential value to small number <= 1e-16 for bad psi-movies
    badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
    badNodesPsisTau = readBadNodesPsisTau(badNodesPsisTaufile)


    # from bad taus (badNodesPsisTau) and split block movies (badNodesPsis)
    if (badNodesPsis.shape[0] == badNodesPsisTau.shape[0]) and (badNodesPsis.shape[1] == badNodesPsisTau.shape[1]):
        badNodesPsis2 = badNodesPsis + badNodesPsisTau
    else:
        badNodesPsis2 = badNodesPsis


    # if badNodesPsis exists

    nodesAllBadPsis = []
    if badNodesPsis2.shape[0] > 0:

        for n in range(badNodesPsis2.shape[0]):  # row has prd numbers, column has psi number so shape is (num_prds,2)
            #remember that badNodePsis has index starting with 1 ??
            badPsis = np.nonzero(badNodesPsis2[n, :] <= -100)[0]

            for k in badPsis:
                if k < NumPsis:
                    nodePot[k, n] = badNodePotVal
                    nodePot[k + NumPsis, n] = badNodePotVal
                else:
                    nodePot[k, n] = badNodePotVal
                    nodePot[k - NumPsis, n] = badNodePotVal

            if len(badPsis) == badNodesPsis2.shape[1]:  # all columns should be bad
                #if len(badPsis)>=badNodesPsis2.shape[1]-1: # all columns, or less one bad ?
                nodesAllBadPsis.append(n)


    #badNodesPsis2 includes  badNodesPsis (split-block movies) + badNodesPsisTau (tau values during optical flow step),
    # so , printing it out as *_bp.txt , the *_of.txt files will have just the badNodesPsisTau
    nodesAllBadPsis = np.array(nodesAllBadPsis)
    print('nodesAllBadPsis', len(nodesAllBadPsis))
    np.savetxt('{}badNodePsis_bp.txt'.format(p.CC_dir), badNodesPsis2, fmt="%d", newline="\n")
    np.savetxt('{}nodesAllBadPsis_bp.txt'.format(p.CC_dir), nodesAllBadPsis + 1, fmt="%d", newline="\n")

    if cc == 2:
        for n in range(nodeStateBP_cc1.shape[0]):  # as nodeStateBP_cc1 is row vector so shape is (num_prds,)
            k = nodeStateBP_cc1[n] - 1  # remember that nodeStateBP_cc1 has index starting with 1 from previous run
            if k < NumPsis:
                nodePot[k, n] = lowNodePotVal
                nodePot[k + NumPsis, n] = lowNodePotVal
            else:
                nodePot[k, n] = lowNodePotVal
                nodePot[k - NumPsis, n] = lowNodePotVal

    print('nodePot.shape:', nodePot.shape, 'edgePot.shape', edgePot.shape)
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Global optimization with Belief propagation
    % Local pairwise measures for the projection direction/psi Topos and movies 
    % are encoded in the undirected probabilistic graphical model as Markov Random Field(MRF) 

    % options for belief propagation iterations
    % options: options for belief propagation iterations
    % options.tol
    % options.verbose
    % options.maximize
    % options.maxIter
    % options.eqnStates 
    '''

    options = dict(maxProduct=BPoptions['maxProduct'],
                   verbose=BPoptions['verbose'],
                   tol=BPoptions['tol'],
                   maxIter=BPoptions['maxIter'],
                   eqnStates=BPoptions['eqnStates'],
                   alphaDamp=BPoptions['alphaDamp'])

    #;%.98;%0.99; %0.99 use damping factor (< 1) when message oscillates and do not converge

    G['anchorNodes'] = anchorNodes

    #nodeOrderType = 'default' # sequential order
    nodeOrderType = 'multiAnchor'

    graphNodeOrder = createNodeOrder(G, anchorNodes, nodeOrderType)
    G['graphNodeOrder'] = graphNodeOrder


    BPalg = createBPalg(G, options)
    BPalg['anchorNodes'] = anchorNodes

    nodeBelief, edgeBelief, BPalg = MRFBeliefPropagation.op(BPalg, nodePot, edgePot)


    nodeBeliefR = nodeBelief

    if enforceBadStateRemoval:
        nodeBeliefR = nodeBelief
        badS = badNodesPsis2 == -100
        print(badS[0, :], np.shape(badS))
        badStates = np.hstack((badS, badS)).T  # FWD + REV states

        nodeBeliefR[badStates] = 0.0



    OptNodeLabels = np.argsort(-nodeBeliefR, axis=0)
    nodeStateBP = OptNodeLabels[0, :]  # %max-marginal
    OptNodeBel = nodeBeliefR[nodeStateBP, range(0, len(nodeStateBP))]




    #%%%%% Determine the Psi's and Senses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print('\nDetermining the psinum and senses from node labels ...')
    nodeStateBP = nodeStateBP + 1  # indexing from 1 as matlab

    psinumsBP, sensesBP = getPsiSensesfromNodeLabels(nodeStateBP, NumPsis)

    psinums_cc = np.zeros((1, G['nNodes']), dtype='int')
    senses_cc = np.zeros((1, G['nNodes']), dtype='int')

    noAnchorCC = G['ConnCompNoAnchor']

    nodesEmptyMeas = []
    for c in noAnchorCC:

        nodesEmptyMeas.append(G['NodesConnComp'][c])

    nodesEmptyMeas = [y for x in nodesEmptyMeas for y in x]
    nodesEmptyMeas = np.array(nodesEmptyMeas)
    print('nodesEmptyMeas:', nodesEmptyMeas)

    psinums_cc[:] = psinumsBP - 1  #python starts with 0
    senses_cc[:] = sensesBP

    psinums_cc = psinums_cc.flatten()
    senses_cc = senses_cc.flatten()
    '''
    if enforceBadStateRemoval:
        # all bad psis set to -1, above condition is redundant will be removed later
        bad_ids = np.nonzero(badNodesPsis2[:,psinums_cc]==-100)[0]
        print('Total bad psinum PDs marked:',np.shape(bad_ids))
        psinums_cc[bad_ids]=-1
        senses_cc[bad_ids]=0
    '''

    # if no measurements for a node,as it was an isolated node
    if len(nodesEmptyMeas) > 0:
        # put psinum/senses value to -1, for the nodes 'nodesEmpty' for which there were no calculations done.
        psinums_cc[nodesEmptyMeas] = -1
        senses_cc[nodesEmptyMeas] = 0

    # if all psi-states for a node was bad
    if len(nodesAllBadPsis) > 0:
        psinums_cc[nodesAllBadPsis] = -1
        senses_cc[nodesAllBadPsis] = 0

    print('Total bad psinum PDs marked:', np.sum(psinums_cc == -1))

    print('psinums_cc', psinums_cc)
    print('senses_cc', senses_cc)

    # If you have manual labels you can %%%% compare here with the manual labels
    compareLabelAcc = 0
    if compareLabelAcc:

        justCompareID = 1  # 1 = files from previous run are same or not, 0 = get accuracy against a manual label
        '''
        from matlab
        recoFile = '../../Results/dataS2/ReCo.mat' # change your path for manual labels
        recoData = loadmat(recoFile)
        psiNumsAll = recoData['psiNumsAll']
        sensesAll = recoData['sensesAll']
        '''
        if not justCompareID:
            # change your path for manual labels
            labelfile = '../../outputs_testvATPase/CC/temp_anchors_20190805-210129.txt'
            recoFileLabel = np.loadtxt(labelfile)

            nodes = recoFileLabel[:, 0].astype(int)
            psinums = recoFileLabel[:, 1].astype(int)
            senses = recoFileLabel[:, 2].astype(int)

            psiNumsAll = (-1) * np.ones((G['nNodes']))
            sensesAll = np.zeros((G['nNodes']))

            #for n in range(0,psiNumsAll.shape[0]):
            #    if i==
            psiNumsAll[nodes - 1] = psinums - 1
            sensesAll[nodes - 1] = senses


            Acc = (sum(((psiNumsAll[:,cc-1] - psinums_cc)==0)*((sensesAll[:,cc-1] - senses_cc)==0)) - sum(psiNumsAll[:,cc-1]==-1))/\
            float(psiNumsAll.shape[0] - sum(psiNumsAll[:,cc-1]==-1))
            print('\nAccuracy: {}'.format(Acc))

            samePsiSensePrds = np.nonzero(
                ((psiNumsAll[:, cc - 1] - psinums_cc) == 0) * ((sensesAll[:, cc - 1] - senses_cc) == 0))[0]
            np.savetxt('samePsiSensePrds.txt', samePsiSensePrds + 1, fmt='%i\t', delimiter='\t')

            diffPsiPrds = np.setdiff1d(range(0, psiNumsAll.shape[0]), samePsiSensePrds)
            np.savetxt('diffPsiPrds.txt', diffPsiPrds + 1, fmt='%i\t', delimiter='\t')

        else:
            # change your path for manual labels
            #ccfilename = '/mnt/Data2/suvrajit/Research/PythonManifoldCode/outputs_cSnu_A5_all/CC/CC_file_somechecked'
            ccfilename = '/mnt/Data2/suvrajit/Research/PythonManifoldCode/outputs_cSnu_A5_all/CC/CC_file_anch38'
            f = open(ccfilename, 'rb')
            data = pickle.load(f)

            psiNumsAll = data['psinums']
            sensesAll = data['senses']
            IDAcc = (sum(((psiNumsAll[cc - 1, :] - psinums_cc) == 0) *
                         ((sensesAll[cc - 1, :] - senses_cc) == 0))) / float(psiNumsAll.shape[1])
            print('\nComparision identity (with a previous set, not accuracy) %: {}'.format(IDAcc))

            samePsiSensePrds = np.nonzero(
                ((psiNumsAll[cc - 1, :] - psinums_cc) == 0) * ((sensesAll[cc - 1, :] - senses_cc) == 0))[0]
            np.savetxt('samePsiSensePrds.txt', samePsiSensePrds + 1, fmt='%i\t', delimiter='\t')

            diffPsiPrds = np.setdiff1d(range(0, psiNumsAll.shape[1]), samePsiSensePrds)
            np.savetxt('diffPsiPrds.txt', diffPsiPrds + 1, fmt='%i\t', delimiter='\t')

    return (nodeStateBP, psinums_cc, senses_cc, OptNodeBel, nodeBelief)
