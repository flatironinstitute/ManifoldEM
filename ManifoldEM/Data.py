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

from ManifoldEM import util, star
from ManifoldEM.params import p


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

    q = util.eul_to_quat(phi, theta, psi, flip)

    return (sh, q, U, V)


def cart2sph(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x) * 180. / math.pi
    theta = math.acos(z / r) * 180. / math.pi  #it was theta
    return (r, phi, theta)


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
