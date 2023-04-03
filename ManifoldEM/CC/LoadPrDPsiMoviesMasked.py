import cv2
import logging
import mrcfile

import numpy as np

from ManifoldEM import annularMask, myio, p
from ManifoldEM.CC import projectMask
'''		
Copyright (c) Columbia University Suvrajit Maji 2020		
Modified:Sept 17,2021
'''

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def getMask2D(prD, maskType, radius):


    if maskType == 'annular':  #annular mask
        N = p.nPix
        diam_angst = p.obj_diam
        diam_pix = diam_angst / p.pix_size
        if radius == None:  # if no input is provided
            N2 = N / 2. - .25 * (N - diam_pix) * 0.30
        else:
            N2 = radius  # also includes radius = 0
        if prD == 0:
            print('Annular mask radius: {} pixels'.format(N2))
        mask = annularMask.op(0, N2, N, N)

    elif maskType == 'volumetric':  #3d volume mask from user-input
        dist_file = p.get_dist_file(prD)
        data = myio.fin1(dist_file)
        PD = data['PD']
        maskFile = p.mask_vol_file

        with mrcfile.open(maskFile) as mrc:
            mask3D = mrc.data

        mask = projectMask.op(mask3D, PD)

    else:
        mask = 1
    '''
    elif maskType=='average2Dmovie':
        mask=2 # do it after reading the movie
    else:
        mask=1
        '''

    return mask


def maskAvgMovie(M):
    # masked being applied to each frame of the movie M
    numFrames = M.shape[0]
    dim = int(np.sqrt(M.shape[1]))
    print('\nnumFrames', numFrames, 'dim', dim)
    M2 = np.resize(M, (numFrames, dim, dim))
    Mavg = np.sum(M2, axis=0)
    mask2D = cv2.adaptiveThreshold(Mavg, np.max(Mavg.flatten()), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                   2)
    maskedM = M * mask2D.flatten('F')  # broadcast to all frames
    # test for first frame
    #plt.imshow(maskedM[1,:].reshape((dim,dim)),cmap='gray')
    #plt.show()
    return maskedM


def findBadNodePsiTau(tau, tau_occ_thresh=0.33):
    quartile_1, quartile_3 = np.percentile(tau, [25, 75])
    iqr = quartile_3 - quartile_1


    ## Sept 2021
    # check if the tau value distribution have more than one narrow ranges far apart
    # this will artifically make the IQR value high giving the illusion of a wide tau
    # distribution. This cases need to be checked and set the IQR to a low value=0.01
    taubins = 50
    tau_h, bin_edges = np.histogram(tau, bins=taubins)  # there are 50 states
    tau_h = np.array(tau_h)
    tau_nz = np.where(tau_h > 0.0)[0].size
    tau_occ = tau_nz / float(taubins)



    # if number of states present in tau values is less than occ_thresh=30%?then there are lot of
    # missing states,
    #tau_occ_thresh = 0.35 # 35% conservative here? # could input through p.py / gui ?

    if (iqr < 0.02) or (tau_occ <= tau_occ_thresh):

        badPsi = 1
    else:
        badPsi = 0

    return badPsi, iqr, tau_occ


def op(prD):

    p.findBadPsiTau = 1  # interface with GUI, p.py
    p.tau_occ_thresh = 0.35  # interface with GUI, p.py
    '''
    useMask = 1 # default
    p.mask_vol_file = ''
    #use mask or not, for movies
    if not p.mask_vol_file:
        # use default annular/circular mask
        maskType = 'annular'
    else:
        useMask = 1
        maskType ='mask3Dprojection'
    '''
    useMask = 0  # default
    if p.opt_mask_type == 0:
        useMask = 0
        maskType = 'None'
        radius = p.opt_mask_param
    elif p.opt_mask_type == 1:
        useMask = 1
        maskType = 'annular'
        radius = p.opt_mask_param
    elif p.opt_mask_type == 2:
        useMask = 1
        maskType = 'volumetric'
        radius = None  # for volumetric we don't need any radius

    psi2_file = p.psi2_file
    NumPsis = p.num_psis

    moviePrDPsis = [None] * NumPsis
    tauPrDPsis = [None] * NumPsis
    badPsis = []
    tauPsisIQR = []
    tauPsisOcc = []
    k = 0
    if useMask:
        # create one mask for a prD
        mask2D = getMask2D(prD, maskType, radius)

    for psinum in range(NumPsis):
        imgPsiFileName = p.get_psi2_file(prD) + f'_psi_{psinum}'
        data_IMG = myio.fin1(imgPsiFileName)
        #IMG1 = data_IMG["IMG1"].T

        IMG1 = data_IMG["IMG1"].T
        tau = data_IMG["tau"]
        #psirec = data_IMG['psirec']
        #psiC1 = data_IMG['psiC1']


        Mpsi = -IMG1

        # checkflip
        if useMask:
            # masked being applied to each frame of the movie M
            if maskType == 'average2Dmovie':
                Mpsi_masked = maskAvgMovie(Mpsi)
            else:
                Mpsi_masked = Mpsi * (mask2D.flatten('F'))  # broadcast to all frames
        else:
            Mpsi_masked = Mpsi

        #Mpsi_masked = rotate_psi(prD,Mpsi_masked)

        moviePrDPsis[psinum] = Mpsi_masked
        tauPrDPsis[psinum] = tau

        if p.findBadPsiTau:
            b, tauIQR, tauOcc = findBadNodePsiTau(tau, p.tau_occ_thresh)
            tauPsisIQR.append(tauIQR)
            tauPsisOcc.append(tauOcc)
            if b:

                badPsis.append(psinum)
                k = k + 1
        else:
            badPsis = []


    return moviePrDPsis, badPsis, tauPrDPsis, tauPsisIQR, tauPsisOcc
