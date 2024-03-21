import multiprocessing
import os

import numpy as np

from numpy import linalg as LA
from functools import partial

from ManifoldEM import myio
from ManifoldEM.params import p
from ManifoldEM.CC.OpticalFlowMovie import SelectFlowVec
from ManifoldEM.util import NullEmitter
from fasthog import hog_from_gradient as histogram_from_gradients


'''
% def CompareOrientMatrix(FlowVecSelA,FlowVecSelB):
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Dec 2017. Modified:Aug 22,2019
  Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''


def HOGOpticalFlowPy(flowVec):
    cell_size = (4, 4)
    cells_per_block = (2, 2)
    n_bins = 9
    signed_orientation = True
    norm_type = 'L2-Hys'

    hog_params = dict(cell_size=cell_size,
                      cells_per_block=cells_per_block,
                      n_bins=n_bins)
    VxDim = flowVec['Vx'].shape
    if len(VxDim) > 2:
        VxStackDim = VxDim[2]

        tempH = []
        for d in range(0, VxStackDim):
            gx = flowVec['Vx'][:, :, d]
            gy = flowVec['Vy'][:, :, d]

            tH = histogram_from_gradients(gx,
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

        gx = flowVec['Vx']
        gy = flowVec['Vy']
        H = histogram_from_gradients(gx,
                                     gy,
                                     cell_size=cell_size,
                                     cells_per_block=cells_per_block,
                                     n_bins=n_bins,
                                     signed=signed_orientation,
                                     norm_type=norm_type)

    return H, hog_params


# Compare how similar two Matrices/Images are.
# TODO: Implement error checking for wrong or, improper inputs
# Check for NaN or Inf outputs , etc.
def CompareOrientMatrix(FlowVecSelA, FlowVecSelB, prds_psinums):
    useNorm = 'l2'
    prD_A = prds_psinums[0]
    psinum_A = prds_psinums[1]
    prD_B = prds_psinums[2]
    psinum_B = prds_psinums[3]

    HOGFA, hog_params = HOGOpticalFlowPy(FlowVecSelA)
    HOGFB, hog_params = HOGOpticalFlowPy(FlowVecSelB)

    # The dimensions of HOGFA and HOGFB should always match given the number of movie blocks created for movie A and B
    # if for some reason the number of blocks for movie A and B are different, then this check is a fail safe to make
    # the code still work
    hogDimA = HOGFA.shape

    hoffset = 1.25
    distHOGAB = []
    distHOGAB_tblock = []
    isBadPsiAB_block = []
    hp = np.ceil(float(hogDimA[0]) / hog_params['cell_size'][0]).astype(int)
    num_hogel_th = np.ceil(0.2 * (hp**2) * hogDimA[2]).astype(int)

    if useNorm == 'l1':
        if len(hogDimA) > 3:

            distHOGAB_tblock = np.zeros((hogDimA[3], 1))
            isBadPsiA_block = np.zeros((hogDimA[3], 1))
            isBadPsiB_block = np.zeros((hogDimA[3], 1))

            for j in range(0, hogDimA[3]):
                if np.count_nonzero(HOGFA[:, :, :, j]) <= num_hogel_th:
                    HOGFA[:, :, :, j] = np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                    isBadPsiA_block[j] = 1

                if np.count_nonzero(HOGFB[:, :, :, j]) <= num_hogel_th:
                    HOGFB[:, :, :, j] = np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                    isBadPsiB_block[j] = 1

                distHOGAB_tblock[j] = sum(abs(HOGFA[:, :, :, j] - HOGFB[:, :, :, j]))

            isBadPsiAB_block = [isBadPsiA_block.T, isBadPsiB_block.T]

        # this should be done after the adjustments of the zero matrix to a matrix with high random numbers
        distHOGAB = sum(abs(HOGFA - HOGFB))

    if useNorm == 'l2':
        if len(hogDimA) > 3:
            distHOGAB_tblock = np.zeros((hogDimA[3], 1))
            isBadPsiA_block = np.zeros((hogDimA[3], 1))
            isBadPsiB_block = np.zeros((hogDimA[3], 1))
            for j in range(0, hogDimA[3]):

                # hog feature matrix difference for A,B HOGFA - HOGFB will be smaller if either of the two matrices are
                # all zeros, so to produce a maximum difference between a normal feature matrix and such zero
                # feature matrix we can add some random numbers with a high value
                if np.count_nonzero(HOGFA[:, :, :, j]) <= num_hogel_th:  # have to check this criteria
                    HOGFA[:, :, :, j] = np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                    isBadPsiA_block[j] = 1

                if np.count_nonzero(HOGFB[:, :, :, j]) <= num_hogel_th:
                    HOGFB[:, :, :, j] = np.random.random(np.shape(HOGFB[:, :, :, j])) + hoffset
                    isBadPsiB_block[j] = 1

                distHOGAB_tblock[j] = LA.norm(HOGFA[:, :, :, j] - HOGFB[:, :, :, j])

            isBadPsiAB_block = [isBadPsiA_block.T, isBadPsiB_block.T]

        # this should be done after the adjustments of the zero matrix to a matrix with high random numbers
        distHOGAB = LA.norm(HOGFA - HOGFB)

    varargout = [distHOGAB, distHOGAB_tblock, isBadPsiAB_block]
    return varargout


def ComparePsiMoviesOpticalFlow(FlowVecSelA, FlowVecSelB, prds_psinums):
    # Analysis of the flow matrix
    psiMovFlowOrientMeasures = dict(Values=[], Values_tblock=[])
    Values, Values_tblock, isBadPsiAB_block = CompareOrientMatrix(FlowVecSelA, FlowVecSelB, prds_psinums)
    psiMovFlowOrientMeasures.update(Values=Values, Values_tblock=Values_tblock)

    return psiMovFlowOrientMeasures, isBadPsiAB_block


def ComputeMeasuresPsiMoviesOpticalFlow(FlowVecSelAFWD, FlowVecSelBFWD, FlowVecSelBREV, prds_psinums):
    psiMovOFMeasuresFWD, isBadPsiAB_blockF = ComparePsiMoviesOpticalFlow(FlowVecSelAFWD, FlowVecSelBFWD, prds_psinums)
    psiMovMFWD = psiMovOFMeasuresFWD['Values']
    psiMovMFWD_tblock = psiMovOFMeasuresFWD['Values_tblock']

    psiMovOFMeasuresREV, isBadPsiAB_blockR = ComparePsiMoviesOpticalFlow(FlowVecSelAFWD, FlowVecSelBREV, prds_psinums)
    psiMovMREV = psiMovOFMeasuresREV['Values']
    psiMovMREV_tblock = psiMovOFMeasuresREV['Values_tblock']

    psiMovieOFmeasures = dict(MeasABFWD=psiMovMFWD,
                              MeasABFWD_tblock=psiMovMFWD_tblock,
                              MeasABREV=psiMovMREV,
                              MeasABREV_tblock=psiMovMREV_tblock)
    return psiMovieOFmeasures, isBadPsiAB_blockF


def ComputeEdgeMeasurePairWisePsiAll(input_data, G, flowVecPctThresh):
    currPrD = input_data[0]
    nbrPrD = input_data[1]
    CC_meas_file = input_data[2]
    edgeNum = input_data[3]

    currentPrDPsiFile = '{}{}'.format(p.CC_OF_file, currPrD)
    nbrPrDPsiFile = '{}{}'.format(p.CC_OF_file, nbrPrD)

    NumPsis = p.num_psi
    # load the data for the current and neighbor prds
    data = myio.fin1(currentPrDPsiFile)
    FlowVecCurrPrD = data['FlowVecPrD']
    data = myio.fin1(nbrPrDPsiFile)
    FlowVecNbrPrD = data['FlowVecPrD']

    nEdges = G['nEdges']

    if len(FlowVecCurrPrD[0]['FWD']['Vx'].shape) > 2:
        numtblocks = FlowVecCurrPrD[0]['FWD']['Vx'].shape[2]
    else:
        numtblocks = 1

    measureOFCurrNbrFWD = np.empty((nEdges, NumPsis, NumPsis))
    measureOFCurrNbrREV = np.empty((nEdges, NumPsis, NumPsis))

    if numtblocks > 1:
        measureOFCurrNbrFWD_tblock = np.empty((nEdges, NumPsis, NumPsis * numtblocks))
        measureOFCurrNbrREV_tblock = np.empty((nEdges, NumPsis, NumPsis * numtblocks))

    psiSelcurrPrD = range(NumPsis)
    psiCandidatesNnbrPrD = range(NumPsis)  # in case psis for currPrD is different from nbrPrD

    badNodesPsisBlock = np.zeros((G['nNodes'], NumPsis))

    for psinum_currPrD in psiSelcurrPrD:
        if FlowVecCurrPrD[psinum_currPrD]['FWD']:  # check if this condition holds for all kind of entries of the dict
            FlowVecCurrPrDFWD = SelectFlowVec(FlowVecCurrPrD[psinum_currPrD]['FWD'], flowVecPctThresh)

        # psi selection candidates for the neighboring prD
        for psinum_nbrPrD in psiCandidatesNnbrPrD:
            if FlowVecNbrPrD[psinum_nbrPrD]['REV']:
                FlowVecNbrPrDFWD = SelectFlowVec(FlowVecNbrPrD[psinum_nbrPrD]['FWD'], flowVecPctThresh)
                FlowVecNbrPrDREV = SelectFlowVec(FlowVecNbrPrD[psinum_nbrPrD]['REV'], flowVecPctThresh)

            prds_psinums = [currPrD, psinum_currPrD, nbrPrD, psinum_nbrPrD]

            FlowVecSelAFWD = FlowVecCurrPrDFWD
            FlowVecSelBFWD = FlowVecNbrPrDFWD
            FlowVecSelBREV = FlowVecNbrPrDREV

            psiMovieOFmeasures, isBadPsiAB_block = ComputeMeasuresPsiMoviesOpticalFlow(
                FlowVecSelAFWD, FlowVecSelBFWD, FlowVecSelBREV, prds_psinums)

            measureOFCurrNbrFWD[edgeNum][psinum_currPrD, psinum_nbrPrD] = psiMovieOFmeasures['MeasABFWD']
            measureOFCurrNbrREV[edgeNum][psinum_currPrD, psinum_nbrPrD] = psiMovieOFmeasures['MeasABREV']

            if numtblocks > 1:
                badNodesPsisBlock[currPrD, psinum_currPrD] = -100 * np.sum(isBadPsiAB_block[0])
                badNodesPsisBlock[nbrPrD, psinum_nbrPrD] = -100 * np.sum(isBadPsiAB_block[1])

            if numtblocks > 1:
                t = psinum_nbrPrD * numtblocks
                print('t', t, 'numtblocks', numtblocks)
                measureOFCurrNbrFWD_tblock[edgeNum][psinum_currPrD, t:t + numtblocks] = np.transpose(
                    psiMovieOFmeasures['MeasABFWD_tblock'])
                measureOFCurrNbrREV_tblock[edgeNum][psinum_currPrD, t:t + numtblocks] = np.transpose(
                    psiMovieOFmeasures['MeasABREV_tblock'])

    measureOFCurrNbrEdge = np.hstack((measureOFCurrNbrFWD[edgeNum], measureOFCurrNbrREV[edgeNum]))

    if numtblocks > 1:
        measureOFCurrNbrEdge_tblock = np.hstack(
            (measureOFCurrNbrFWD_tblock[edgeNum], measureOFCurrNbrREV_tblock[edgeNum]))
    else:
        measureOFCurrNbrEdge_tblock = []

    myio.fout1(CC_meas_file,
               measureOFCurrNbrEdge=measureOFCurrNbrEdge,
               measureOFCurrNbrEdge_tblock=measureOFCurrNbrEdge_tblock,
               badNodesPsisBlock=badNodesPsisBlock)


# changed Nov 30, 2018, S.M.
# here N is a list of (edge) numbers
def divide1(N, G):
    ll = []

    for e in N:
        currPrD = G['Edges'][e, 0]
        nbrPrD = G['Edges'][e, 1]
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file, e, currPrD, nbrPrD)
        ll.append([currPrD, nbrPrD, CC_meas_file, e])

    return ll


def op(G, nodeEdgeNumRange, *argv):
    multiprocessing.set_start_method('fork', force=True)

    p.load()

    nodeRange = nodeEdgeNumRange[0]
    edgeNumRange = nodeEdgeNumRange[1]
    if len(edgeNumRange) == 0:
        edgeNumRange = range(G['nEdges'])
    numberofJobs = len(nodeRange) + len(edgeNumRange)
    flowVecPctThresh = p.opt_movie['flowVecPctThresh']

    offset = 0
    if argv:
        progress5 = argv[0]
    else:
        progress5 = NullEmitter()

    #extract info for psi selection/sense of ref and psi candidates for nbr
    input_data = divide1(edgeNumRange, G)  # changed Nov 30, 2018, S.M.

    if argv:
        offset = numberofJobs - len(input_data)
        progress5.emit(int((offset / float(numberofJobs)) * 99))

    if p.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in enumerate(input_data):
            ComputeEdgeMeasurePairWisePsiAll(datai, G, flowVecPctThresh)
            if argv:
                progress5.emit(int(((offset + i) / float(numberofJobs)) * 99))
    else:
        with multiprocessing.Pool(processes=p.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(
                    partial(ComputeEdgeMeasurePairWisePsiAll, G=G, flowVecPctThresh=flowVecPctThresh),
                    input_data)):
                progress5.emit(int(((offset + i) / float(numberofJobs)) * 99))

    p.save()
