import numpy as np
import pandas as pd

from ManifoldEM import util
from ManifoldEM.params import p

'''
Copyright (c) Columbia University Sonya Hanson 2018
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Evan Seitz 2021
'''


def write_star(star_file, traj_file, df):
    p.load()

    with open(star_file, 'w') as text_file:
        text_file.write(
            '\ndata_ \n \nloop_ \n \n_rlnImageName #1 \n_rlnAnglePsi #2 \n_rlnAngleTilt #3 \n_rlnAngleRot #4 \n_rlnDetectorPixelSize #5 \n_rlnMagnification #6 \n'
        )
        for i in range(len(df)):
            text_file.write('%s@%s %s %s %s %s %s\n' %
                            (i + 1, traj_file, df.psi[i], df.theta[i], df.phi[i], p.pix_size, 10000.0))
            # Note: DetectorPixelSize and Magnification required by relion_reconstruct; 10000 used here such that we can always put in the user-defined pixel size...
            # ...since it may be obtained via calibration (see user manual); since Pixel Size = Detector Pixel Size [um] / Magnification --> [Angstroms]


'''
This parse_star function is from version 0.1 of pyem by Daniel Asarnow at UCSF
'''


def parse_star(starfile, skip, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'rU') as f:
        for l in f:
            if ln < skip:
                ln += 1
            else:
                if l.startswith("_rln"):
                    foundheader = True
                    lastheader = True
                    if keep_index:
                        head = l.rstrip()
                    else:
                        head = l.split('#')[0].rstrip().lstrip('_')
                    headers.append(head)
                else:
                    lastheader = False
                if foundheader and not lastheader:
                    break
                ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None)
    star.columns = headers

    return star


def parse_star_optics(starfile, keep_index=False):  #added by E. Seitz -- 10/23/21
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'rU') as f:
        for l in f:
            if l.startswith("_rln"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.rstrip()
                else:
                    head = l.split('#')[0].rstrip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None, nrows=1)
    star.columns = headers

    return (star, ln + 1)


def get_align_data(align_star_file, flip):
    relion_old = True
    with open(align_star_file, 'r') as f:
        for line in f:
            if line.startswith("data_optics"):
                relion_old = False
                break

    if relion_old:
        skip = 0
        df = parse_star(align_star_file, skip, keep_index=False)
        df0 = df
    else:
        print('RELION Optics Group found.')
        df0, skip = parse_star_optics(align_star_file, keep_index=False)
        df = parse_star(align_star_file, skip, keep_index=False)

    try:
        p.EkV = float(df0['rlnVoltage'].values[0])
        p.Cs = float(df0['rlnSphericalAberration'].values[0])
        p.AmpContrast = float(df0['rlnAmplitudeContrast'].values[0])
    except:
        print('missing microscope parameters')
        exit(1)

    try:
        U = df['rlnDefocusU'].values
        V = df['rlnDefocusV'].values
    except:
        print("missing defocus")
        exit(1)

    if 'rlnOriginX' in df.columns and 'rlnOriginY' in df.columns:
        shx = df['rlnOriginX'].values
        shy = df['rlnOriginY'].values
    elif 'rlnOriginXAngst' in df.columns and 'rlnOriginYAngst' in df.columns:
        shx = df['rlnOriginXAngst'].values / p.pix_size
        shy = df['rlnOriginYAngst'].values / p.pix_size
    else:
        print(f"Warning: missing relion origin data in {align_star_file}")
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
