import copy
import multiprocessing
import operator
import os
import shutil

import numpy as np

from ManifoldEM import myio
from ManifoldEM.data_store import data_store
from ManifoldEM.params import params
from ManifoldEM.util import NullEmitter
from ManifoldEM.CC import OpticalFlowMovie, LoadPrDPsiMoviesMasked


# changed Nov 30, 2018, S.M.
# here N is a list of (node) numbers
def _construct_input_data(R):
    ll = []
    for prD in R:
        CC_OF_file = f'{params.CC_OF_file}{prD}'
        if os.path.exists(CC_OF_file) and os.path.getsize(CC_OF_file):
            continue
        ll.append([CC_OF_file, prD])
    return ll


'''
function ComputeOpticalFlowPrDAll
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: May 2018. Modified:Aug 16,2019
'''


def stackDicts(a, b, op=operator.concat):
    op = lambda x, y: np.dstack((x, y))
    mergeDict = dict(a.items() + b.items() + [(k, op(a[k], b[k])) for k in set(b) & set(a)])
    return mergeDict


def ComputePsiMovieOpticalFlow(Mov, opt_movie, prds_psinums):
    OFvisualPrint = [opt_movie['OFvisual'], opt_movie['printFig']]
    Labels = ['FWD', 'REV']

    computeOF = 1
    blockSize_avg = 5  # how many frames will used for normal averaging
    currPrD = prds_psinums[0]
    psinum_currPrD = prds_psinums[1]
    prd_psinum = [currPrD, psinum_currPrD]

    MFWD = copy.deepcopy(Mov)
    numFrames = Mov.shape[0]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute the Optical Flow vectors for each movie
    # For complicated motion involving some rotation component, the 2d movie can be misleading if we get the
    # optical flow vector added over the entire movie , so we might split the movie into blocks and compare
    # the vectors separtely , with splitmovie = 1, this is experimental now.

    # at present the stacking of the dictionary for two blocks has been checked, for others needs to be verified
    splitmovie = False
    FlowVecFWD = []
    FlowVecREV = []
    if computeOF:
        if splitmovie:
            # number of frames in each blocks
            numBlocks_split = 3
            overlapFrames = 0  #12

            blockSize_split = np.round(
                np.float(numFrames + (numBlocks_split - 1) * overlapFrames + 1) / (numBlocks_split)).astype(int)

            # In case we fix blockSize, it should be noted that the numBlocks will be different for different
            # blocksize and overlap values
            # Also, one extra block is used in case there is 1 or 2 frames left over after frameEnd is close to
            # numFrames and a new block is created with overlapping frames till frameEnd = numFrames
            # TO DO:better handling of this splitting into overlapping blocks
            for b in range(0, numBlocks_split):
                frameStart = max(0, b * (blockSize_split - overlapFrames))
                frameEnd = min(b * (blockSize_split - overlapFrames) + blockSize_split - 1, numFrames)

                if numFrames - frameEnd < 5:
                    frameEnd = numFrames
                # check this criteria
                if frameEnd - frameStart > 0:
                    blockMovieFWD = MFWD[frameStart:frameEnd, :]

                    FlowVecFblock = OpticalFlowMovie.op(blockMovieFWD, prd_psinum, blockSize_avg,
                                                        Labels[0] + '-H' + str(b), OFvisualPrint)
                    if b == 0:
                        FlowVecFWD = copy.deepcopy(FlowVecFblock)
                    else:
                        FlowVecFWD = stackDicts(FlowVecFWD, FlowVecFblock)

                    # blockMovieFWD is used but due to label of 'REV', the negative vectors will be used after computing the FWD vectors
                    # If FWD vectors are provided, then reverse flow vectors are not going to be recomputed but will
                    # be obtained by reversing the FWD vectors (-Vx,-Vy)
                    # use FlowVecFblock as it is just one block, FlowVecFWD for multiple blocks has multidimensional Vx and Vy--stacked
                    FlowVecRblock = OpticalFlowMovie.op(blockMovieFWD, prd_psinum, blockSize_avg,
                                                        Labels[1] + '-H' + str(b), OFvisualPrint, FlowVecFblock)

                    if b == 0:
                        FlowVecREV = copy.deepcopy(FlowVecRblock)
                    else:
                        FlowVecREV = stackDicts(FlowVecREV, FlowVecRblock)

                if frameEnd == numFrames:
                    break
        else:
            FlowVecFWD = OpticalFlowMovie.op(MFWD, prd_psinum, blockSize_avg, Labels[0], OFvisualPrint)

            # MFWD is used but due to label of 'REV', the negative vectors will be used after getting the FWD vectors
            FlowVecREV = OpticalFlowMovie.op(MFWD, prd_psinum, blockSize_avg, Labels[1], OFvisualPrint, FlowVecFWD)

    FlowVec = dict(FWD=FlowVecFWD, REV=FlowVecREV)

    return FlowVec


def ComputeOptFlowPrDPsiAll1(input_data):
    CC_OF_file = input_data[0]
    currPrD = input_data[1]
    FlowVecPrD = np.empty(params.num_psi, dtype=object)
    psiSelcurrPrD = range(params.num_psi)

    # load movie and tau param first
    moviePrDPsi, badPsis, tauPrDPsis, tauPsisIQR, tauPsisOcc = LoadPrDPsiMoviesMasked.op(currPrD)

    badPsis = np.array(badPsis)
    CC_dir_temp = '{}temp/'.format(params.CC_dir)

    os.makedirs(CC_dir_temp, exist_ok=True)

    badNodesPsisTaufile_pd = '{}badNodesPsisTauFile_PD_{}'.format(CC_dir_temp, currPrD)

    badNodesPsisTau = np.copy(badPsis)
    NodesPsisTauIQR = tauPsisIQR
    NodesPsisTauOcc = tauPsisOcc
    NodesPsisTauVals = tauPrDPsis

    myio.fout1(badNodesPsisTaufile_pd,
               badNodesPsisTau=badNodesPsisTau,
               NodesPsisTauIQR=NodesPsisTauIQR,
               NodesPsisTauOcc=NodesPsisTauOcc,
               NodesPsisTauVals=NodesPsisTauVals)

    # calculate OF for each psi-movie
    for psinum_currPrD in psiSelcurrPrD:
        IMGcurrPrD = moviePrDPsi[psinum_currPrD]

        prds_psinums = [currPrD, psinum_currPrD]
        FlowVecPrDPsi = ComputePsiMovieOpticalFlow(IMGcurrPrD, params.opt_movie, prds_psinums)
        FlowVecPrD[psinum_currPrD] = FlowVecPrDPsi

    CC_OF_file = '{}'.format(CC_OF_file)
    myio.fout1(CC_OF_file, FlowVecPrD=FlowVecPrD)


# If computing for a specified set of nodes, then call the function with nodeRange
def op(node_edge_num_range, *argv):
    params.load()
    multiprocessing.set_start_method('fork', force=True)

    node_range = node_edge_num_range[0]
    edge_num_range = node_edge_num_range[1]
    numberofJobs = len(node_range) + len(edge_num_range)
    input_data = _construct_input_data(node_range)

    if params.find_bad_psi_tau:
        # initialize and write to file badpsis array
        offset_OF_files = len(node_range) - len(input_data)
        if offset_OF_files == 0:  # offset_OF_files=0 when no OF files were generated
            bad_nodes_psis_taufile = '{}badNodesPsisTauFile'.format(params.CC_dir)
            if os.path.exists(bad_nodes_psis_taufile):
                os.remove(bad_nodes_psis_taufile)

        G = data_store.get_prds().neighbor_graph_pruned
        bad_nodes_psis_tau = np.zeros((G['nNodes'], params.num_psi)).astype(int)
        nodes_psis_tau_IQR = np.zeros((G['nNodes'], params.num_psi)) + 5.  # any positive real number > 1.0 outside tau range
        # tau range is [0,1.0], since a zero or small tau value by default means it will be automatically assigned
        # as a bad tau depending on the cut-off
        nodes_psis_tau_occ = np.zeros((G['nNodes'], params.num_psi))
        nodes_psis_tau_vals = [[None]] * G['nNodes']

        # the above variables are initialized at the start and also at resume of CC step
        # and used later for combining the individual bad tau PD files

        # but make sure the intialized variables are written out to the file only at the start
        # and not during resume of CC step
        if offset_OF_files == 0:
            myio.fout1(bad_nodes_psis_taufile,
                       badNodesPsisTau=bad_nodes_psis_tau,
                       NodesPsisTauIQR=nodes_psis_tau_IQR,
                       NodesPsisTauOcc=nodes_psis_tau_occ,
                       NodesPsisTauVals=nodes_psis_tau_vals)

    if argv:
        offset = len(node_range) - len(input_data)
        progress5 = argv[0]
        progress5.emit(int((offset / float(numberofJobs)) * 99))
    else:
        progress5 = NullEmitter()
        offset = 0

    if params.ncpu == 1:  # avoids the multiprocessing package
        for i, datai in enumerate(input_data):
            ComputeOptFlowPrDPsiAll1(datai)
            progress5.emit(int(((offset + i) / float(numberofJobs)) * 99))
    else:
        with multiprocessing.Pool(processes=params.ncpu) as pool:
            for i, _ in enumerate(pool.imap_unordered(ComputeOptFlowPrDPsiAll1, input_data)):
                progress5.emit(int(((offset + i) / float(numberofJobs)) * 99))

    # for now individual files were written and are being combined here
    if params.find_bad_psi_tau:
        CC_dir_temp = '{}temp/'.format(params.CC_dir)

        # if CC_dir_temp exists and is non-empty  combine the individual files again
        if os.path.exists(CC_dir_temp) and len(os.listdir(CC_dir_temp)) > 0:
            for currPrD in node_range:
                badNodesPsisTaufile_pd = '{}badNodesPsisTauFile_PD_{}'.format(CC_dir_temp, currPrD)
                dataR = myio.fin1(badNodesPsisTaufile_pd)

                badPsis = dataR['badNodesPsisTau']  # based on a specific tau-iqr cutoff in LoadPrDPsiMoviesMasked
                # but we actually use the raw iqr values to get a histogram of all iqr across all PDs to get the better cutoff later.
                tauPsisIQR = dataR['NodesPsisTauIQR']
                tauPsisOcc = dataR['NodesPsisTauOcc']
                tauPrDPsis = dataR['NodesPsisTauVals']
                if len(badPsis) > 0:
                    bad_nodes_psis_tau[currPrD, np.array(badPsis)] = -100
                nodes_psis_tau_IQR[currPrD, :] = tauPsisIQR
                nodes_psis_tau_occ[currPrD, :] = tauPsisOcc
                nodes_psis_tau_vals[currPrD] = tauPrDPsis

            bad_nodes_psis_taufile = '{}badNodesPsisTauFile'.format(params.CC_dir)
            myio.fout1(bad_nodes_psis_taufile,
                       badNodesPsisTau=bad_nodes_psis_tau,
                       NodesPsisTauIQR=nodes_psis_tau_IQR,
                       NodesPsisTauOcc=nodes_psis_tau_occ,
                       NodesPsisTauVals=nodes_psis_tau_vals)

            rem_temp_dir = False
            if rem_temp_dir:
                # remove the temp directory if rem_temp_dir=1, or manually delete later
                print('Removing temp directory', CC_dir_temp)
                if os.path.exists(CC_dir_temp):
                    shutil.rmtree(CC_dir_temp)
