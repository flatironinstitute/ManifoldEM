"""Load image stack/volume/etc data files.

Loads input data files and populates the global parameters with relevant derived values. Also
calculates the graph structure according to the currently set low and high threshold values.

Copyright (c) Columbia University Hstau Liao 2018
Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Suvrajit Maji 2019
Copyright (c) Flatiron Institute Robert Blackwell 2023
"""
import math
import csv

import numpy as np

from ManifoldEM import S2tessellation, myio, FindCCGraph, util, p, star
from ManifoldEM.quaternion import qMult_bsx


def get_from_relion(align_star_file, flip):
    relion_old = True
    with open(align_star_file, 'r') as f:
        for line in f:
            if line.startswith("data_optics"):
                relion_old = False
                break

    if relion_old:
        skip = 0
        df = star.parse_star(align_star_file, skip, keep_index=False)
        try:
            U = df['rlnDefocusU'].values
            V = df['rlnDefocusV'].values
        except:
            print("missing defocus")
            exit(1)
        if 'rlnOriginX' in df.columns and 'rlnOriginY' in df.columns:
            shx = df['rlnOriginX'].values
            shy = df['rlnOriginY'].values
        else:
            shx = U * 0.
            shy = shx
        sh = (shx, shy)
        try:
            phi = np.deg2rad(df['rlnAngleRot'].values)
            theta = np.deg2rad(df['rlnAngleTilt'].values)
            psi = np.deg2rad(df['rlnAnglePsi'].values)
        except:
            print("missing Euler angles")
            exit(1)
        try:
            p.EkV = float(df['rlnVoltage'].values[0])
            p.Cs = float(df['rlnSphericalAberration'].values[0])
            p.AmpContrast = float(df['rlnAmplitudeContrast'].values[0])
        except:
            print('missing microscope parameters')
            exit(1)

    else:
        print('RELION Optics Group found.')
        df0, skip = star.parse_star_optics(align_star_file, keep_index=False)
        try:
            p.EkV = float(df0['rlnVoltage'].values[0])
            p.Cs = float(df0['rlnSphericalAberration'].values[0])
            p.AmpContrast = float(df0['rlnAmplitudeContrast'].values[0])
        except:
            print('missing microscope parameters')
            exit(1)

        df = star.parse_star(align_star_file, skip, keep_index=False)
        try:
            U = df['rlnDefocusU'].values
            V = df['rlnDefocusV'].values
        except:
            print("missing defocus")
            exit(1)
        if 'rlnOriginX' in df.columns and 'rlnOriginY' in df.columns:
            shx = df['rlnOriginX'].values
            shy = df['rlnOriginY'].values
        else:
            shx = U * 0.
            shy = shx
        sh = (shx, shy)
        try:
            phi = np.deg2rad(df['rlnAngleRot'].values)
            theta = np.deg2rad(df['rlnAngleTilt'].values)
            psi = np.deg2rad(df['rlnAnglePsi'].values)
        except:
            print("missing Euler angles")
            exit(1)

    qz, qy, qzs = util.eul_to_quat(phi, theta, psi, flip)
    q = qMult_bsx(qzs, qMult_bsx(qy, qz))

    return (sh, q, U, V)


def parse_spider(filename: str):
    """
    Parse a SPIDER DOC file.

    Parameters
    ----------
    filename : str
               Name of the file

    Returns
    -------
    np.array(dtype=float64)
        Table of values in file
    """
    # Copyright (c) Columbia University Hstau Liao 2018 (python version)
    table = []
    with open(filename, 'r') as fin:
        p.num_part = 0
        for line in fin:
            line1 = line.strip()
            words = line1.split()
            if words[0].find(';') == -1:
                p.num_part += 1
                words = [float(x) for x in words]
                table.append(words)
    table = np.array(table)
    # skip the second column
    table = np.hstack((table[:, 0].reshape(-1, 1), table[:, 2:]))
    return table


def get_q(align_param_file, phiCol, thetaCol, psiCol, flip):
    # read the angles
    align = parse_spider(align_param_file)
    phi = np.deg2rad(align[:, phiCol])
    theta = np.deg2rad(align[:, thetaCol])
    psi = np.deg2rad(align[:, psiCol])
    qz, qy, qzs = util.eul_to_quat(phi, theta, psi, flip)
    q = qMult_bsx(qzs, qMult_bsx(qy, qz))
    return q


def get_df(align_param_file, dfCol):
    # read df
    align = parse_spider(align_param_file)
    if len(dfCol) == 1:
        df = align[:, dfCol]
    if len(dfCol) == 2:
        df = (align[:, dfCol[0]] + align[:, dfCol[1]]) / 2

    return df


def get_shift(align_param_file, shx_col, shy_col):
    # read the x-y shifts
    align = parse_spider(align_param_file)
    sh = (align[:, shx_col] * 0, align[:, shy_col] * 0)
    return sh


def cart2sph(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x) * 180. / math.pi
    theta = math.acos(z / r) * 180. / math.pi  #it was theta
    return (r, phi, theta)


def genColorConnComp(G):
    numConnComp = len(G['NodesConnComp'])

    nodesColor = np.zeros((G['nNodes'], 1), dtype='int')
    for i in range(numConnComp):
        nodesCC = G['NodesConnComp'][i]
        nodesColor[nodesCC] = i

    return nodesColor


def write_angles(ang_file, color, S20, full, NC):
    with open(ang_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter='\t')
        csvwriter.writerow(("PrD", "theta", "phi", "psi", "x", "y", "z", "ClusterID"))

        if full == 1:  # already thresholded S20
            L = range(0, S20.shape[1])
        else:  # full S20, still need to take the correct half
            mid = S20.shape[1] // 2
            NC1 = NC[:int(mid)]
            NC2 = NC[int(mid):]
            if len(NC1) >= len(NC2):  # first half of S2
                L = range(0, mid)
            else:
                L = range(mid, int(S20.shape[1]))  # second half of S2

        for idx, prD in enumerate(L):
            x, y, z = S20[0:3, prD]
            r, phi, theta = cart2sph(x, y, z)
            clusterID = color[prD][0] if full else 0

            csvwriter.writerow((idx + 1, theta, phi, 0, x, y, z, clusterID))



def op(align_param_file):
    p.load()
    visual = False

    sh, q, U, V = get_from_relion(align_param_file, flip=True)
    df = (U + V) / 2
    # double the number of data points by augmentation
    q = util.augment(q)
    df = np.concatenate((df, df))

    CG1, CG, nG, S2, S20_th, S20, NC = S2tessellation.op(q, p.ang_width, p.PDsizeThL, visual, p.PDsizeThH)
    # CG1: list of lists, each of which is a list of image indices within one PD
    # CG: thresholded version of CG1
    # nG: approximate number of tessellated bins
    # S2: cartesian coordinates of each of particles' angular position on S2 sphere
    # S20_th: thresholded version of S20
    # S20: cartesian coordinates of each bin-center on S2 sphere
    # NC: list of occupancies of each PD

    # copy ref angles S20 to file
    myio.fout1(p.tess_file, CG1=CG1, CG=CG, nG=nG, q=q, df=df, S2=S2, S20=S20_th, sh=sh, NC=NC)

    p.numberofJobs = len(CG)
    p.save()

    if p.resProj == 0 and (np.shape(CG)[0] > 2):
        G, Gsub = FindCCGraph.op()
        nodesColor = genColorConnComp(G)

        write_angles(p.ref_ang_file, nodesColor, S20_th, 1, NC)  # to PrD_map.txt (thresh bins)
        write_angles(p.ref_ang_file1, nodesColor, S20, 0, NC)  # to PrD_map1.txt (all bins)
