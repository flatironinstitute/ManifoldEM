import os

import numpy as np

from ManifoldEM import myio, DMembeddingII
from ManifoldEM.params import p
'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2020 (python version) 
'''


def op(orig_zip, new_zip, PrD):
    print('Initiating re-embedding...')
    dist_file = p.get_dist_file(PrD)
    psi_file = p.get_psi_file(PrD)
    eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, PrD + 1)
    data = myio.fin1(dist_file)
    D = data['D']
    data = myio.fin1(psi_file)
    posPath = data['posPath']
    ind = data['ind']
    D = D[posPath][:, posPath]  # D now contains the orig distances

    # Py3 update -- E.Seitz, 2021:
    origX, origY = zip(*orig_zip)  #unpack points
    newX, newY = zip(*new_zip)  #unpack points
    orig = np.stack((origX, origY), axis=1)
    new = np.stack((newX, newY), axis=1)
    c = np.in1d(orig.view('i,i').reshape(-1), new.view('i,i').reshape(-1))
    cR = np.reshape(c, (int(np.shape(c)[0] / 2), 2))
    posPathInd = np.where(cR[:, 0])[0]  #the ordered indices of 2D-coordinates contained in both lists

    D1 = D[posPathInd][:, posPathInd]  # distances of the new points only
    k = D1.shape[0]
    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op(D1, k, p.tune, 60000)  #updated 9/11/21
    posPath = posPath[posPathInd]  # update posPath
    lamb = lamb[lamb > 0]

    if os.path.exists(eig_file):
        os.remove(eig_file)

    for i in range(len(lamb) - 1):
        with open(eig_file, "a") as file:  #updated 9/11/21
            file.write("%d\t%.5f\n" % (i + 1, lamb[i + 1]))

    myio.fout1(psi_file, lamb=lamb, psi=psi, sigma=sigma, mu=mu, posPath=posPath, ind=ind)

    # remove the existing NLSA and movies etc, so that new ones can be created
    for psinum in range(p.num_psis):
        psi2_file = p.get_psi2_file(PrD) + f'_psi_{psinum}'
        if os.path.exists(psi2_file):
            os.remove(psi2_file)

    ca_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, PrD + 1)
    if os.path.exists(ca_file):
        os.remove(ca_file)
