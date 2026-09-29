import numpy as np
from nptyping import NDArray, Shape, Float64
import pandas as pd

from ManifoldEM.params import params
from ManifoldEM.util import eul_to_quat

'''
Copyright (c) Columbia University Sonya Hanson 2018
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Evan Seitz 2021
'''


def write_star(star_file, traj_file, phi, theta, psi, pixel_size):
    with open(star_file, 'w') as text_file:
        text_file.write(
            '\ndata_ \n \nloop_ \n \n_rlnImageName #1 \n_rlnAnglePsi #2 \n_rlnAngleTilt #3 \n_rlnAngleRot #4 \n_rlnDetectorPixelSize #5 \n_rlnMagnification #6 \n'
        )
        for i in range(len(phi)):
            text_file.write('%s@%s %s %s %s %s %s\n' %
                            (i + 1, traj_file, psi[i], theta[i], phi[i], pixel_size, 10000.0))
            # Note: DetectorPixelSize and Magnification required by relion_reconstruct; 10000 used here such that we can always put in the user-defined pixel size...
            # ...since it may be obtained via calibration (see user manual); since Pixel Size = Detector Pixel Size [um] / Magnification --> [Angstroms]



def parse_star(starfile, skip):
    """
    Parses a STAR file and returns the data as a pandas DataFrame.

    Parameters
    ----------
    starfile : str
        The path to the STAR file to be parsed.
    skip : int
        The number of lines to skip at the beginning of the file before starting.
        to look for headers. This is useful for skipping comments or metadata at the top of the file.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the data from the STAR file, with columns named according to the
        headers found in the file.

    Notes
    -----
    - The function first scans the file to find headers (lines starting with "_rln"). It records
      these headers and determines where the data section starts.
    - Column names are the header names without the leading underscore and the trailing
      "#index" (e.g. "_rlnVoltage #1" becomes "rlnVoltage").
    - After identifying the headers and the start of the data section, the function reads the
      data into a pandas DataFrame, using the headers as column names.
    - This function is specifically tailored for STAR files used in cryo-EM data processing and
      may not be suitable for STAR files with a significantly different format.
    - This parse_star function is from version 0.1 of pyem by Daniel Asarnow at UCSF
    """

    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
        for l in f:
            if ln < skip:
                ln += 1
            else:
                if l.startswith("_rln"):
                    foundheader = True
                    lastheader = True
                    headers.append(l.split('#')[0].rstrip().lstrip('_'))
                else:
                    lastheader = False
                if foundheader and not lastheader:
                    break
                ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter=r'\s+', header=None)
    star.columns = headers

    return star


def parse_star_optics(starfile: str):
    """
    Parses the optics section of a STAR file and returns the data as a pandas DataFrame.

    Parameters
    ----------
    starfile : str
        The path to the STAR file to be parsed.

    Returns
    -------
    tuple
        pd.DataFrame
            A pandas DataFrame containing the first row of data from the optics section
            of the STAR file, with columns named according to the headers found in the file.
        int
            The line number where the data section ends, useful for further parsing.

    Notes
    -----
    - The function scans the file for headers starting with "_rln". These headers define the columns
      of the optics section.
    - Column names are the header names without the leading underscore and the trailing
      "#index" (e.g. "_rlnVoltage #1" becomes "rlnVoltage").
    - The function reads only the first row of data under the headers into a DataFrame, assuming
      that the optics section contains a single set of parameters.
    - This function is useful for extracting optics-related metadata from STAR files used in
      cryo-EM data processing.
    - This implementation was added by E. Seitz -- 10/23/21
    """

    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
        for l in f:
            if l.startswith("_rln"):
                foundheader = True
                lastheader = True
                headers.append(l.split('#')[0].rstrip().lstrip('_'))
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter=r'\s+', header=None, nrows=1)
    star.columns = headers

    return (star, ln + 1)


def get_align_data(align_star_file: str) -> tuple[tuple[NDArray[Shape["*"], Float64],
                                                                    NDArray[Shape["*"], Float64]],
                                                              NDArray[Shape["4,*"], Float64],
                                                              NDArray[Shape["*"], Float64],
                                                              NDArray[Shape["*"], Float64],
                                                              ]:
    """
    Extracts alignment data and microscope parameters from a RELION STAR file.

    Parameters
    ----------
    align_star_file : str
        Path to the STAR file containing alignment and microscope parameters.

    Returns
    -------
    tuple
        tuple
            A tuple of numpy arrays (shx, shy) representing the shifts in `X` and `Y`.
        np.ndarray
            A 4xN numpy array of quaternions representing the rotations in (phi, ux, uy, uz) form.
        np.ndarray
            A numpy array containing the defocus `U` values.
        np.ndarray
            A numpy array containing the defocus `V` values.

    Notes:
    - The function first checks if the STAR file is in the old or new RELION format by looking
      for the "data_optics" section.
    - It then parses the optics and particles sections accordingly using `parse_star` and
      `parse_star_optics` functions to extract the required data.
    - Microscope parameters such as voltage, spherical aberration, and amplitude contrast are
      extracted from the optics section.
    - Alignment parameters including defocus values, shifts, and Euler angles are extracted from
      the particles section.
    - Shifts are adjusted based on available columns and pixel size, and Euler angles are
      converted to quaternions.
    - The function is designed to work with RELION STAR files and may need adjustments for
      compatibility with other formats or versions.
    """

    relion_old = True
    with open(align_star_file, 'r') as f:
        for line in f:
            if line.startswith("data_optics"):
                relion_old = False
                break

    if relion_old:
        skip = 0
        df = parse_star(align_star_file, skip)
        df0 = df
    else:
        print('RELION Optics Group found.')
        df0, skip = parse_star_optics(align_star_file)
        df = parse_star(align_star_file, skip)

    try:
        params.ms_kilovolts = float(df0['rlnVoltage'].values[0])
        params.ms_spherical_aberration = float(df0['rlnSphericalAberration'].values[0])
        params.ms_amplitude_contrast_ratio = float(df0['rlnAmplitudeContrast'].values[0])
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
        shx = df['rlnOriginXAngst'].values / params.ms_pixel_size
        shy = df['rlnOriginYAngst'].values / params.ms_pixel_size
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

    q = eul_to_quat(phi, theta, psi)

    return (sh, q, U, V)
