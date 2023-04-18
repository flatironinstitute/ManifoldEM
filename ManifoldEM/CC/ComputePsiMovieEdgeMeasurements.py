import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as ndifilt

from scipy import signal
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import fsolve, curve_fit
from skimage import filters
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ManifoldEM import myio, p, FindCCGraphPruned
from ManifoldEM.CC import ComputeOpticalFlowPrDAll, ComputeMeasureEdgeAll
'''
Copyright (c) Columbia University Suvrajit Maji 2019
Modified:Sept 21,2021
'''

# this rescaling function should ensure to keep the exp(-M) values within a certain range as to prevent
# numerical overflow/underflow
# do it for all values across all edges, to check for outliers and relative edge values after scaling
# are comparable in this way
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


def findThreshHist(X, nbins, method=1):
    # this is still experimental
    # would work if the data values are sort of bi-modal distribution

    def relativeMaxMin(data, nPointsWindow, sigmaWindow):
        window = signal.general_gaussian(nPointsWindow, p=1, sig=sigmaWindow)
        dataFiltered = signal.fftconvolve(data, window, 'same')
        dataFiltered *= np.average(data) / np.average(dataFiltered)
        indexMax = signal.argrelmax(dataFiltered)[0]
        indexMin = signal.argrelmin(dataFiltered)[0]
        return dataFiltered, indexMax, indexMin

    def findIntersectionOfFuncs(funce, funcg, popt, x0):
        p1 = popt[0]
        p2 = popt[1]
        p3 = popt[2]
        p4 = popt[3]
        p5 = popt[4]
        sol_root = fsolve(lambda x: funce(x, p1, p2) - funcg(x, p3, p4, p5), x0)
        return sol_root

    def separateHist(X, labels, cluster_centers, bedges, tl):
        id0 = np.argmin(cluster_centers)
        return np.max(X[labels == id0])

    # histogram
    h, bedges = np.histogram(X, bins=nbins)
    bctrs = bedges[:-1] + np.diff(bedges) / 2.0

    if method == 0 or method == 1 or method == 4 or method == 'all':
        # 1. kmeans
        kmeans = KMeans(n_clusters=2).fit(X)
        t_thresh_k = separateHist(X, kmeans.labels_, kmeans.cluster_centers_.T, bedges, 'kmeans')
        if method == 0 or method == 'all':
            print('0. Kmeans.threshold:', t_thresh_k, ', centers:', kmeans.cluster_centers_.T)

        t = t_thresh_k.copy()

    if method == 1 or method == 'all':
        # 2. find_peaks -- valleys for 'inverted' data
        # somehow argrelextrema did not produce all correct extremas for different data
        # so sticking with find_peaks
        # TO DO. If there are multiple valleys found, then choose the one in between the two high-peaks found using
        # yfilt and not -yfilt
        # or other methods such as kmeans (method=1) ... just have to eliminate the valley (inverted peaks) indexes
        # outside the the two peak values...(assumption is there are only two high peaks...for multiple high peaks,
        # there will be different criteria)
        yfilt = ndifilt.uniform_filter1d(h, 4, mode='nearest')
        _, xbins = np.histogram(h, nbins - 1)
        figy = plt.figure(figsize=(12, 6))
        plt.plot(xbins, yfilt)
        yfigfile = os.path.join(p.CC_dir, 'yfilt')
        figy.savefig(yfigfile + '.png')

        pk_inv, _ = find_peaks(-yfilt, prominence=np.max(yfilt) // 3)
        print('find peaks:', pk_inv, bctrs[pk_inv])
        _, pkmax, pkmin = relativeMaxMin(h, 21, 3)
        print('rel min peaks:', pkmin, bctrs[pkmin])
        if not any(pk_inv):  # pk_inv is empty for some reason, find_peaks did not work
            pk = pkmin
        else:
            pk = np.union1d(pk_inv, pkmin)

        print('all peaks:', pk, bctrs[pk])
        valleys = bctrs[pk]
        indx = np.where(
            np.logical_and(valleys >= np.min(kmeans.cluster_centers_), valleys <= np.max(kmeans.cluster_centers_)))
        t_thresh_p = valleys[indx]
        print('1. find peaks threshold:', t_thresh_p)
        # if there are multiple peaks/valleys, choose the one between the centers found by other reliable methods
        # such as kmeans below
        t = t_thresh_p.copy()

    if method == 2 or method == 'all':
        # 3. otsu
        t_thresh_o = filters.threshold_otsu(X)
        print('2. Otsu.threshold:', t_thresh_o)
        t = t_thresh_o

    if method == 3 or method == 'all':
        # works best if the multimodal distributions are all mixture of gaussians
        #4. GMM
        gmm = GaussianMixture(n_components=2, covariance_type="full")
        gmm.fit(X)

        # TODO: if mixture does not seem to contain both gaussians
        # so, fit separate mixture of two different distributions (e.g. exponential + gaussian)
        mthresh = 0.2  # check this based on stdevs ??
        if np.max(np.sqrt(gmm.covariances_)) < mthresh:
            print('... Gmm: individual covariance did not work well (mixture is possibly not all gaussians)... using '
                  'tied'
                  ' covariance for gmm.')
            gmm = GaussianMixture(n_components=2, covariance_type="tied")
            gmm.fit(X)

        glabels = gmm.predict(X)
        t_thresh_g = separateHist(X, glabels, gmm.means_.T, bedges, 'gmm')
        print('3. Gmm.threshold:', t_thresh_g, 'centers:', gmm.means_.T)
        t = t_thresh_g.copy()

    if method == 4 or method == 'all':
        # 5. This for only if we have exp+gauss (given the data) ...
        # a. If the data has only mixture of gaussians, then use gauss+gauss curve fit,
        # in fact gmm(method=4) would work fine in that case and curve fit would not be required.
        # b. Uses filtered h.
        # c. Also uses kmeans results (method=1) as intial guess for gauss center
        # Note: if for some reason , the curve-fitting fails, we use the default kmeans
        kmeans = KMeans(n_clusters=2).fit(X)

        def funce(x, ea, eb):
            return ea * np.exp(-eb * x)

        def funcg(x, ga, gb, gc):
            return ga * np.exp(-((x - gb) / gc)**2)

        def func(x, ea, eb, ga, gb, gc, f):
            return funce(x, ea, eb) + funcg(x, ga, gb, gc) + f

        # this p0 is for func only
        p0 = [np.max(h), 3.0, np.max(h) / 1.5, np.max(kmeans.cluster_centers_), 1.0, 0.0]

        xdata = bctrs
        ydata = savgol_filter(h, 7, 2)

        try:
            # fitting may not always converge or work
            popt, pcov = curve_fit(func, xdata, ydata, p0, bounds=(-np.inf, np.inf))
            t_thresh_c = findIntersectionOfFuncs(funce, funcg, popt, 0.0)
            print('4. Curve_fit intersection.threshold:', t_thresh_c, ', centers:', kmeans.cluster_centers_.T)
            t = t_thresh_c.copy()
        except:
            print('4. curve-fitting failed..., using kmeans instead.')
            # in that case just use the kmeans values (method=1)?
            t_thresh_c = t_thresh_k.copy()
            print('Kmeans.threshold:', t_thresh_c, ', centers:', kmeans.cluster_centers_.T)
            t = t_thresh_c.copy()

    if method == 'all':
        t = np.append(t_thresh_k, np.append(t_thresh_p, [t_thresh_o, t_thresh_g]))
        t = np.append(t, t_thresh_c)

    print('\nDistribution cutoff(s):', t)
    return t, np.max(h), yfilt


def checkBadPsis(trash_list, tau_occ_thresh=0.35):
    # Oct 2020, this is still experimental
    # It would be good to interface this part also with the GUI , to visually check the cut-off selected
    # for the bad tau iqr distribution and occupancy ...
    # Check if there are significant bad PDs(>10) after Optical Flow computations of the psi-movies,
    # if yes then prune those bad nodes. The nodes are not removed from the graph but the edges are modified
    # just update the graph G with new edge connections

    # the bad psis are also checked during BP for setting tiny node potentials for bad node states
    # it is there in case, the threshold needs to be modified to exclude or include more bad psi-movies
    #
    # the graph edges can be pruned here
    badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
    dataR = myio.fin1(badNodesPsisTaufile)

    badNodesPsisTau = dataR['badNodesPsisTau']  # this was pre-calculated using some tau-cutoff, here we are update it
    TausMat_IQR = dataR['NodesPsisTauIQR']
    TausMat_Occ = dataR['NodesPsisTauOcc']
    TausMat = TausMat_IQR  #

    print('Using all psi-tau values across all nodes to find the tau(iqr) distribution-cutoff')
    TausAll = TausMat.flatten()
    X = TausAll.reshape(-1, 1)

    #tau_thresh = findThreshHist(X)

    nbins = 50  # we could use optimal bin finding methods such as 'fd','scott' etc,
    #plt.savefig('tau_iqrhist_cutoff.png')

    Allmethods = ['K-means', 'find_peaks', 'Otsu', 'GMM', 'Curve-fit Intersection']
    # choose cutoff method type
    # 0. Kmeans
    # 1: find_peaks
    # 2: Otsu
    # 3. GMM (this works really well when the individual distributions are all gaussians,
    #        otherwise GMM will have issues)
    # 4. Curve fitting with Kmeans

    method = 'all'  # integer between 0 to 4 or 'all'
    numAllMethods = len(Allmethods)

    if method == 'all':
        methods = range(numAllMethods)
        print('Method:', Allmethods)
    else:
        methods = []
        methods.append(method)
        print('Method:', Allmethods[method])

    cutoff, hmax, yfilt = findThreshHist(X, nbins, method=method)

    if cutoff.size > 1:  # how do we compare and choose the cut-off when using multiple methods?
        # just choose the min, max, mean , median , etc. oro compare
        # the cut-off lines visually and just choose the best one ?
        #tau_thresh = np.median(cutoff)
        #best_id = 1 #? visually compare [0...4]
        #tau_thresh = cutoff[best_id]
        tau_thresh = np.median(cutoff[~np.isnan(cutoff)])
    else:
        tau_thresh = cutoff.copy()

    print('Tau(iqr) distribution cutoff selected:', tau_thresh)
    if tau_thresh.size > 0:
        #bad_idx = (TausMat<tau_thresh)
        bad_idx_iqr = (TausMat_IQR < tau_thresh)
        print('Tau occupancy cutoff preset:', tau_occ_thresh)
        bad_idx_occ = (TausMat_Occ < tau_occ_thresh)  # test, temp Sept 2021
        bad_idx = np.logical_or(bad_idx_iqr, bad_idx_occ)
    else:
        bad_idx = []
    badNodesPsisTau = np.zeros(np.shape(TausMat))
    badNodesPsisTau[bad_idx] = -100

    extra = dict(badNodesPsisTau_of=badNodesPsisTau)
    dataR.update(extra)


    #previously was generating a separate file *_of but now just adding an extra variable to the same file
    #badNodesPsisTau_of which should used at BP step
    badNodesPsisTaufile_of = '{}'.format(badNodesPsisTaufile)
    myio.fout1(badNodesPsisTaufile_of, **dataR)

    # sort of redundant to set -100 and not bad_idx directly but for consistency, we leave it like this for now
    NumbadPsis = np.sum(badNodesPsisTau == -100, axis=1)
    NumPsis = badNodesPsisTau.shape[1]
    nodesAllBadPsis = np.nonzero(NumbadPsis == NumPsis)[0]
    num_nodesAllBadPsis = len(nodesAllBadPsis)
    print('Number of trash PDs detected using auto tau-cutoff:', num_nodesAllBadPsis)

    np.savetxt('{}NodeTauPsis_of.txt'.format(p.CC_dir), TausMat_IQR, fmt="%f", newline="\n")
    np.savetxt('{}NodeTauPsisOcc_of.txt'.format(p.CC_dir), TausMat_Occ, fmt="%f", newline="\n")
    np.savetxt('{}badNodePsis_of.txt'.format(p.CC_dir), badNodesPsisTau, fmt="%d", newline="\n")
    np.savetxt('{}nodesAllBadPsis_of.txt'.format(p.CC_dir), nodesAllBadPsis + 1, fmt="%d", newline="\n")

    if nodesAllBadPsis.shape[0] > 0:
        print('nodesAllBadPsis', nodesAllBadPsis)
        trash_list[nodesAllBadPsis] = True

    return trash_list, num_nodesAllBadPsis


def op(G, nodeRange, edgeNumRange, *argv):

    nodeEdgeNumRange = [nodeRange, edgeNumRange]

    # Step 1. Compute Optical Flow Vectors
    # Save the optical flow vectors for each psi-movie of individual projection direction
    if p.getOpticalFlow:
        print('\n1.Now computing optical flow vectors for all (selected) PrDs...\n')
        # Optical flow vectors for each psi-movies of each node are saved to disk
        ComputeOpticalFlowPrDAll.op(nodeEdgeNumRange, *argv)

    # check for bad PDs found based on bad tau values
    trash_list = p.get_trash_list()
    p.tau_occ_thresh = 0.35  # interface with GUI, p.py
    tau_occ_thresh = p.tau_occ_thresh

    # take the already existing trash_list and update it
    trash_list_chk, num_nodesAllBadPsis = checkBadPsis(trash_list, tau_occ_thresh)

    # trash_list_chk will be used inside the following pruned graph creation if p.use_pruned_graph =1
    # p.trash_list = trash_list_chk
    # FindCCGraphPruned uses p.trash_list to create the pruned graph
    CC_graph_file_pruned = '{}_pruned'.format(p.CC_graph_file)

    p.use_pruned_graph = 0  # interface with gui
    if p.use_pruned_graph:
        #Step 2a. June 2020
        ### Check if there are significant bad PDs(>10 or 5 ?) after Optical Flow computations of the psi-movies,
        # if yes then prune those bad nodes. The nodes are not removed from the graph but the edges are modified
        # just update the graph G with new edge connections
        #the bad psis are also checked during BP for setting tiny node potentials for bad node states
        # the graph edges can be pruned here

        num_bad_nodes_prune_cutoff = 5
        print('Pruning the graph G if there are more than {} bad nodes'.format(num_bad_nodes_prune_cutoff))

        # update the p.trash_list
        p.set_trash_list(trash_list_chk)
        if num_nodesAllBadPsis > num_bad_nodes_prune_cutoff:
            if not os.path.exists(CC_graph_file_pruned):
                G, Gsub = FindCCGraphPruned.op(CC_graph_file_pruned)
            else:
                print('Using a previously pruned graph.')
                data = myio.fin1(CC_graph_file_pruned)
                G = data['G']
                Gsub = data['Gsub']
            numConnComp = len(G['NodesConnComp'])

            anchorlist = [a[0] - 1 for a in p.anch_list]  # we need labels with 0 index to compare with the node labels in G, Gsub
            nodelCsel = []
            edgelCsel = []
            # this list keeps track of the connected component (single nodes included) for which no anchor was provided
            connCompNoAnchor = []
            for i in range(numConnComp):
                nodesGsubi = Gsub[i]['originalNodes']
                edgelistGsubi = Gsub[i]['originalEdgeList']

                if any(x in anchorlist for x in nodesGsubi) or len(nodesGsubi) > 1:
                    nodelCsel.append(nodesGsubi.tolist())
                    edgelCsel.append(edgelistGsubi[0])
                else:
                    connCompNoAnchor.append(i)

            if len(connCompNoAnchor) > 0:
                print('There are {} connected components with no anchors assigned. You can choose anchors for them ' \
                      'after the edge measurements are done, and re-run only the BP'.format(len(connCompNoAnchor)))

            nodeRange = np.sort([y for x in nodelCsel for y in x])  #flatten list another way?
            edgeNumRange = np.sort([y for x in edgelCsel for y in x])  #flatten list another way?

            nodeEdgeNumRange = [nodeRange, edgeNumRange]

            data = myio.fin1(CC_graph_file_pruned)
            extra = dict(nodeRange=nodeRange, edgeNumRange=edgeNumRange, ConnCompNoAnchor=connCompNoAnchor)
            data.update(extra)
            myio.fout1(CC_graph_file_pruned, **data)

    # Step 2. Compute the pairwise edge measurements
    # Save individual edge measurements
    if p.getAllEdgeMeasures:
        print('\n2.Now computing pairwise edge-measurements...\n')
        # measures for creating potentials later on
        # edgeMeasures files for each edge (pair of nodes) are saved to disk
        if argv:
            ComputeMeasureEdgeAll.op(G, nodeEdgeNumRange, argv[0])
        else:
            ComputeMeasureEdgeAll.op(G, nodeEdgeNumRange)

    # Step 3. Extract the pairwise edge measurements
    # to be used for node-potential and edge-potential calculations
    print('\n3.Reading all the edge measurements from disk...')
    # load the measurements file for each edge separately

    # in case there are some nodes/edges for which we do not want to calculate the measures, the number of edges and
    # max edge indices may not match, so use the full G.nEdges as the size of the edgeMeasures. The edges which are not
    # calculated will remain as empty
    print('Edges', G['nEdges'])
    edgeMeasures = np.empty((G['nEdges']), dtype=object)
    edgeMeasures_tblock = np.empty((G['nEdges']), dtype=object)
    badNodesPsisBlock = np.zeros((G['nNodes'], p.num_psis))

    for e in edgeNumRange:
        currPrD = G['Edges'][e, 0]
        nbrPrD = G['Edges'][e, 1]

        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file, e, currPrD, nbrPrD)
        data = myio.fin1(CC_meas_file)
        measureOFCurrNbrEdge = data['measureOFCurrNbrEdge']
        measureOFCurrNbrEdge_tblock = data['measureOFCurrNbrEdge_tblock']
        bpsi = data['badNodesPsisBlock']
        badNodesPsisBlock = badNodesPsisBlock + bpsi
        edgeMeasures[e] = measureOFCurrNbrEdge
        edgeMeasures_tblock[e] = measureOFCurrNbrEdge_tblock

    # This rescaling step is to prevent underflow/overflow, should be checked if does not work
    scaleRange = [5, 45]
    edgeMeasures = rescale_linear(edgeMeasures, edgeNumRange, scaleRange)

    return edgeMeasures, edgeMeasures_tblock, badNodesPsisBlock
