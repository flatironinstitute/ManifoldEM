import logging
import mrcfile
import os

import numpy as np

from nptyping import NDArray, Shape, Float64

from scipy.ndimage import shift
from scipy.fftpack import ifftshift, fft2, ifft2

from ManifoldEM import myio, projectMask
from ManifoldEM.params import p
from ManifoldEM.core import annularMask
from ManifoldEM.quaternion import q2Spider
from ManifoldEM.util import rotate_fill
"""
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) UWM, Peter Schwander 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
"""

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

version = 'getDistanceCTF_local9, V 1.0'

def create_grid(N: int) -> NDArray[Shape["*,*"], Float64]:
    if N <= 0:
        _logger.error('non-positive image size')
        _logger.exception('non-positive image size')
        raise ValueError

    if N % 2 == 1:
        a = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
    else:
        a = np.arange(-N / 2, N / 2)
    X, Y = np.meshgrid(a, a)

    Q = (1. / (N / 2.)) * np.sqrt(X**2 + Y**2)

    return Q


def create_filter(filter_type, NN, Qc, Q):
    if filter_type == 'Gauss':
        G = np.exp(-(np.log(2) / 2.) * (Q / Qc)**2)
    elif filter_type == 'Butter':
        G = np.sqrt(1. / (1 + (Q / Qc)**(2 * NN)))
    else:
        _logger.error('%s filter is unsupported' % (filter_type))
        _logger.exception('%s filter is unsupported' % (filter_type))
        raise ValueError

    return G


def calc_avg_pd(q, nS):
    # Calculate average projection directions (from matlab code)
    PDs = 2 * np.vstack((q[1, :] * q[3, :] - q[0, :] * q[2, :], q[0, :] * q[1, :] + q[2, :] * q[3, :],
                         q[0, :]**2 + q[3, :]**2 - np.ones((1, nS)) / 2.0))

    return PDs


def get_psi(q, PD, iS):
    # Quaternion approach
    s = -(1 + PD[2]) * q[3, iS] - PD[0] * q[1, iS] - PD[1] * q[2, iS]
    c = (1 + PD[2]) * q[0, iS] + PD[1] * q[1, iS] - PD[0] * q[2, iS]
    psi = 2 * np.arctan(s / c)  # note that the Psi are in the interval [-pi,pi]

    return (psi, s, c)


def psi_ang(PD):
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0])
    Qr = Qr / np.sqrt(np.sum(Qr**2))
    _, _, psi = q2Spider(Qr)

    psi = np.mod(psi, 2 * np.pi) * (180 / np.pi)
    return psi


def get_wiener1(CTF1):
    SNR = 5
    wiener_dom = 0.
    for i in range(CTF1.shape[0]):
        wiener_dom = wiener_dom + CTF1[i, :, :]**2

    wiener_dom = wiener_dom + 1. / SNR

    return (wiener_dom)


def ctemh_cryoFrank(k, spherical_aberration: float, defocus: float, electron_energy: float,
                    gauss_env_halfwidth: float, amplitude_contrast_ratio: float):
    """
    from Kirkland, adapted for cryo (EMAN1) by P. Schwander
    Version V 1.1
    Copyright (c) UWM, Peter Schwander 2010 MATLAB version

    Copyright (c) Columbia University Hstau Liao 2018 (python version)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Here, the damping envelope is characterized by a single parameter B (gauss_env)
    see J. Frank
    spherical_abberation (Cs) in mm
    defocus (df) in Angstrom, a positive value is underfocus
    electron_energy in keV
    gauss_env (B) in A^-2
    amplitude_constrast_ratio from alignment file

    Note: we assume |k| = s
    """
    spherical_aberration *= 1.0e7
    mo = 511.0
    hc = 12.3986
    wav = (2 * mo) + electron_energy
    wav = hc / np.sqrt(wav * electron_energy)
    w1 = np.pi * spherical_aberration * wav * wav * wav
    w2 = np.pi * wav * defocus
    k2 = k * k
    sigm = gauss_env_halfwidth / np.sqrt(2 * np.log(2))
    wi = np.exp(-k2 / (2 * sigm**2))
    wr = (0.5 * w1 * k2 - w2) * k2  # gam = (pi/2)Cs lam^3 k^4 - pi lam df k^2

    y = (np.sin(wr) - amplitude_contrast_ratio * np.cos(wr)) * wi
    return y


def op(input_data, filter_par, img_file_name, image_offsets, n_particles_tot, avg_only, relion_data):
    """
    Calculates squared Euclidian distances for snapshots in similar
    projection directions. Includes CTF correction of microspope.
    Version with conjugates, effectively double number of data points

    Input parameters
    input_data
                 input_data['indices']      Global image indexes
                 input_data['quats']        Rotation quaternions of all images, 4xN
                 input_data['defocus']      Defocus values of all images
                 input_data['dist_file']    Output file for results
    filter_par   filter Gaussian width [pixel]
                 filterPar['Qc']         Nyquist cutoff frequency
                 filterPar['type']       Filter type 'Butter' or 'Gauss'
                 filterPar['N']          Filter order (for 'Butter' only)
    image_offsets  Image origins (from star files, usually. aka "sh")
    img_file_name  Image file with all raw images
    avg_only       Skip calculation of distances

    Uses the following microscope data from global params:
        Cs         Spherical aberration [mm]
        EkV        Acceleration voltage [kV]
        gaussEnv   Gaussian damping envelope [A^-1]
        nPix       lateral pixel count
        dPix       Pixel size [A]
    """
    indices = input_data['indices']
    quats = input_data['quats']
    defocus = input_data['defocus']
    out_file = input_data['dist_file']

    n_particles = indices.shape[0]  # size of bin; ind are the indexes of particles in that bin
    # auxiliary variables
    n_pix = p.nPix

    # initialize arrays
    psis = np.nan * np.ones((n_particles, 1))  # psi angles
    numerator = np.nan * np.ones((n_particles, 1))
    denominator = np.nan * np.ones((n_particles, 1))
    # different types of averages of aligned particles of the same view
    img_avg = np.zeros((n_pix, n_pix))  # simple average
    img_avg_flip = np.zeros((n_pix, n_pix))  # average of phase-flipped particles
    img_all_flip = np.zeros((n_particles, n_pix, n_pix))  # all the averages of phase-flipped particles
    img_all = np.zeros((n_particles, n_pix, n_pix))  #

    flattened_images = np.zeros((n_pix**2, n_particles))  # each row is a flatten image
    fourier_images = complex(0) * np.ones((n_particles, n_pix, n_pix))  # each (i,:,:) is a Fourier image
    CTF = np.zeros((n_particles, n_pix, n_pix))  # each (i,:,:) is the CTF
    distances = np.zeros((n_particles, n_particles))  # distances among the particles in the bin

    msk = annularMask(0, n_pix / 2., n_pix, n_pix)

    # read images with conjugates
    img_labels = np.zeros(n_particles, dtype=int)
    for i_part in range(n_particles):
        if indices[i_part] < n_particles_tot / 2:  # first half data set; i.e., before augmentation
            indiS = int(indices[i_part])
            img_labels[i_part] = 1
        else:  # second half data set; i.e., the conjugates
            indiS = int(indices[i_part] - n_particles_tot / 2)
            img_labels[i_part] = -1
            # matlab version: y[:,iS] = m.Data(ind(iS)).y
        start = n_pix**2 * indiS * 4
        if not relion_data:  # spider data
            tmp = np.memmap(img_file_name, dtype='float32', offset=start, mode='r', shape=(n_pix, n_pix))
            # store each flatted image in y
            tmp = tmp.T  # numpy mapping is diff from matlab's
        else:  # relion data
            tmp = mrcfile.mmap(img_file_name, 'r')
            tmp.is_image_stack()
            tmp = tmp.data[indiS]
            shi = (image_offsets[1][indiS] - 0.5, image_offsets[0][indiS] - 0.5)
            tmp = shift(tmp, shi, order=3, mode='wrap')
        if indices[i_part] >= n_particles_tot / 2:  # second half data set
            tmp = np.flipud(tmp)
        # normalizing
        backg = tmp * (1 - msk)
        try:
            tmp = (tmp - backg.mean()) / backg.std()
        except:
            pass
        # store each flatted image in y
        flattened_images[:, i_part] = tmp.flatten('F')

    # create grid for filter G
    Q = create_grid(n_pix)
    G = create_filter(filter_par['type'], filter_par['N'], filter_par['Qc'], Q)
    G = ifftshift(G)
    # filter each image in the bin
    for i_part in range(n_particles):
        img = flattened_images[:, i_part].reshape(-1, n_pix).transpose()
        img = ifft2(fft2(img) * G).real
        flattened_images[:, i_part] = img.real.flatten('F')

    # Calculate average projection directions
    PDs = calc_avg_pd(quats, n_particles)
    # reference PR is the average
    PD = np.sum(PDs, 1)
    # make it a unit vector
    PD = PD / np.linalg.norm(PD)

    # use volumetric mask, April 2020
    if p.mask_vol_file:
        with mrcfile.open(p.mask_vol_file) as mrc:
            mask3D = mrc.data
        msk2 = projectMask.op(mask3D, PD)
    else:
        msk2 = 1

    # psi_p angle for in-plane rotation alignment
    psi_p = psi_ang(PD)
    # looping for all the images in the bin
    for i_part in range(n_particles):
        # Get the psi angle
        psi, s, c = get_psi(quats, PD, i_part)

        # this happens only for a rotation of pi about an axis perpendicular to the projection direction
        if np.isnan(psi):
            psi = 0.
        psis[i_part] = psi  # save image rotations
        denominator[i_part] = c  # save denominator
        numerator[i_part] = s  # save nominator

        # inplane align the images
        img = flattened_images[:, i_part].reshape(-1, n_pix).transpose() * msk  # convert to matlab convention prior to rotation
        img = rotate_fill(img, -(180 / np.pi) * psi)
        img = rotate_fill(img, -psi_p)

        # CTF info
        tmp = ctemh_cryoFrank(Q / (2 * p.pix_size), p.Cs, defocus[i_part], p.EkV, p.gaussEnv, p.AmpContrast)

        CTF[i_part, :, :] = ifftshift(tmp)  # tmp should be in matlab convention

        CTFtemp = CTF[i_part, :, :]
        # Fourier transformed #April 2020, with vol mask msk2, used for distance calc D
        fourier_images[i_part, :, :] = fft2(img * msk2)

        img_flip = ifft2(np.sign(CTFtemp) * fourier_images[i_part, :, :])  # phase-flipped
        img_all_flip[i_part, :, :] = img_flip.real  # taking all the phase-flipped images
        img_avg_flip = img_avg_flip + img_flip.real  # average of all phase-flipped images
        img_all[i_part, :, :] = img
    del flattened_images

    # use wiener filter
    img_avg = 0
    wiener_dom = -get_wiener1(CTF)
    for i_part in range(n_particles):
        img = img_all[i_part, :, :]
        img_f = fft2(img)  #.reshape(dim, dim)) T only for matlab
        CTF_i = CTF[i_part, :, :]
        img_f_wiener = img_f * (CTF_i / wiener_dom)
        img_avg = img_avg + ifft2(img_f_wiener).real

    # plain and phase-flipped averages
    # April 2020, msk2 = 1 when there is no volume mask
    img_avg = img_avg * msk2 / n_particles
    img_avg_flip = img_avg_flip.real * msk2 / n_particles

    if not avg_only:
        fourier_images = fourier_images.reshape(n_particles, n_pix**2)
        CTF = CTF.reshape(n_particles, n_pix**2)

        CTFfy = CTF.conj() * fourier_images
        distances = np.dot((abs(CTF)**2), (abs(fourier_images)**2).T)
        distances = distances + distances.T - 2 * np.real(np.dot(CTFfy, CTFfy.conj().transpose()))

    img_all_intensity = np.mean(img_all_flip**2, axis=0)

    myio.fout1(out_file, D=distances, ind=indices, q=quats, df=defocus, CTF=CTF, imgAll=img_all, msk2=msk2, PD=PD, PDs=PDs, Psis=psis,
               imgAvg=img_avg, imgAvgFlip=img_avg_flip, imgLabels=img_labels, Dnom=denominator, Nom=numerator,
               imgAllIntensity=img_all_intensity, version=version, avg_only=avg_only, relion_data=relion_data)
