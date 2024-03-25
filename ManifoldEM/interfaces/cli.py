import os
if not 'OMP_NUM_THREADS' in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

import sys
from argparse import ArgumentParser
import ManifoldEM
from ManifoldEM.params import params, ProjectLevel

def get_parser():
    parser = ArgumentParser(
        prog="manifold-cli",
        description="Command-line interface for ManifoldEM package",
    )
    parser.add_argument('-n', '--ncpu', type=int, default=1)
    subparsers = parser.add_subparsers(help=None, dest="command")

    def add_relevant_params(subparser, level: ProjectLevel, prefix: str = ""):
        annotated_params = params.get_params_for_level(level)
        for param, (paramtype, paraminfo) in annotated_params.items():
            if paraminfo.user_param:
                subparser.add_argument(f"--{param}", metavar=paramtype.__name__.upper(), type=paramtype, help=f'{prefix}{paraminfo.description}')


    init_parser = subparsers.add_parser("init", help="0: Initialize new project")
    init_parser.add_argument('-p', "--project-name", type=str, metavar="STR", help="Name of project to create", required=True)
    init_parser.add_argument('-v', "--avg-volume", type=str, metavar="FILEPATH", default="")
    init_parser.add_argument('-a', "--alignment", type=str, metavar="FILEPATH", default="")
    init_parser.add_argument('-i', "--image-stack", type=str, metavar="FILEPATH", default="")
    init_parser.add_argument('-m', "--mask-volume", type=str, metavar="FILEPATH", default="")
    init_parser.add_argument('-s', "--pixel-size", type=float, metavar="FLOAT", required=True)
    init_parser.add_argument('-d', "--diameter", type=float, metavar="FLOAT", required=True)
    init_parser.add_argument('-r', "--resolution", type=float, metavar="FLOAT", required=True)
    init_parser.add_argument('-x', "--aperture-index", type=int, metavar="INT", default=1)
    init_parser.add_argument('-o', '--overwrite', action='store_true',
                             help="Replace existing project with same name automatically")
    for level in ProjectLevel:
        add_relevant_params(init_parser, level, f"[{level.name}] ")

    threshold_parser = subparsers.add_parser("threshold", help="1: Set upper/lower thresholds for principal direction detection")
    threshold_parser.add_argument("input_file", type=str)
    add_relevant_params(threshold_parser, ProjectLevel.BINNING)

    distance_parser = subparsers.add_parser("calc-distance", help="2: Calculate S2 distances")
    distance_parser.add_argument("input_file", type=str)
    distance_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(distance_parser, ProjectLevel.CALC_DISTANCE)

    manifold_analysis_parser = subparsers.add_parser("manifold-analysis", help="4: Initial embedding")
    manifold_analysis_parser.add_argument("input_file", type=str)
    manifold_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(manifold_analysis_parser, ProjectLevel.MANIFOLD_ANALYSIS)

    psi_analysis_parser = subparsers.add_parser("psi-analysis", help="5: Analyze images to get psis")
    psi_analysis_parser.add_argument("input_file", type=str)
    psi_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(psi_analysis_parser, ProjectLevel.PSI_ANALYSIS)

    nlsa_movie_parser = subparsers.add_parser("nlsa-movie", help="6: Create 2D psi movies")
    nlsa_movie_parser.add_argument("input_file", type=str)
    nlsa_movie_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(nlsa_movie_parser, ProjectLevel.NLSA_MOVIE)

    cc_parser = subparsers.add_parser("find-ccs", help="7: Find conformational coordinates")
    cc_parser.add_argument("input_file", type=str)
    add_relevant_params(cc_parser, ProjectLevel.FIND_CCS)

    el_parser = subparsers.add_parser("energy-landscape", help="8: Calculate energy landscape")
    el_parser.add_argument("input_file", type=str)
    add_relevant_params(el_parser, ProjectLevel.ENERGY_LANDSCAPE)

    traj_parser = subparsers.add_parser("trajectory", help="9: Calculate trajectory")
    traj_parser.add_argument("input_file", type=str)
    add_relevant_params(traj_parser, ProjectLevel.TRAJECTORY)

    utility_parser = subparsers.add_parser("utility", help="Utility functions")
    utility_subparsers = utility_parser.add_subparsers(help=None, dest="command")

    mrcs2mrc_parser = utility_subparsers.add_parser("mrcs2mrc",
                                                    help="Convert output of trajectory step from mrcs to mrc [requires working relion install in PATH]")
    mrcs2mrc_parser.add_argument("input_file", type=str)

    denoise_parser = utility_subparsers.add_parser("denoise", help="Denoise output of mrcs2mrc postprocessing step")
    denoise_parser.add_argument("input_file", type=str)
    denoise_parser.add_argument("-k", "--window_size", type=int, metavar="INT", default=5, help="Kernel/window size")
    denoise_parser.add_argument("-f", "--frame", type=int, metavar="INT", default=5, help="Beginning and ending frames affected")
    denoise_parser.add_argument("--filter", type=str, metavar="STR", default="Gaussian", help="Filter type: {Gaussian, Median}")

    return parser


def load_state(args):
    if args.command == "init":
        return
    fname_front = args.input_file.split('params_', 1)[1]
    fname_sans = os.path.splitext(fname_front)[0]
    params.project_name = fname_sans

    params.load(args.input_file)

    params.ncpu = args.ncpu
    if args.command == "threshold":
        params.prd_thres_low = args.low
        params.prd_thres_high = args.high
    if hasattr(args, "path_width"):
        if args.path_width is None:
            return
        if args.path_width < 1 or args.path_width > 5:
            print("path-width argument must be on the interval [1, 5]")
            sys.exit(1)
        params.width_1D = args.path_width

    params.save()


def init(args):
    import shutil
    from ManifoldEM.util import get_image_width_from_stack

    params.project_name = args.project_name
    proj_file = f'params_{params.project_name}.toml'

    if os.path.isfile(proj_file) or os.path.isdir(params.out_dir):
        response = 'y' if args.overwrite else None
        while response not in ('y', 'n'):
            response = input("Project appears to exist. Overwrite? y/n\n").lower()
        if response == 'n':
            print("Aborting")
            return 1
        print("Removing previous project")
        if os.path.isdir(params.out_dir):
            shutil.rmtree(params.out_dir)

    params.avg_vol_file = os.path.expanduser(args.avg_volume)
    params.align_param_file = os.path.expanduser(args.alignment)
    params.img_stack_file = os.path.expanduser(args.image_stack)
    params.mask_vol_file = os.path.expanduser(args.mask_volume)

    params.ms_pixel_size = args.pixel_size
    params.particle_diameter = args.diameter
    params.ms_estimated_resolution = args.resolution
    params.aperture_index = args.aperture_index
    params.is_relion_data = args.alignment.endswith('.star')

    params.ms_num_pixels = get_image_width_from_stack(params.img_stack_file)

    params.create_dir()
    params.save()


def _parse_prd_list(prd_list: str):
    if prd_list:
        return [int(i) for i in prd_list.split(',')]

    return None


def threshold(args):
    params.prd_thres_low = args.low
    params.prd_thres_high = args.high
    params.project_level = ProjectLevel.BINNING
    params.save()


def calc_distance(args):
    from ManifoldEM import calc_distance
    prd_list = _parse_prd_list(args.prds)
    calc_distance.op(prd_list)


def manifold_analysis(args):
    from ManifoldEM import manifold_analysis
    prd_list = _parse_prd_list(args.prds)
    manifold_analysis.op(prd_list)


def psi_analysis(args):
    from ManifoldEM import psi_analysis
    prd_list = _parse_prd_list(args.prds)
    psi_analysis.op(prd_list)


def nlsa_movie(args):
    from ManifoldEM import nlsa_movie
    prd_list = _parse_prd_list(args.prds)
    nlsa_movie.op(prd_list)


def find_conformational_coordinates(_):
    from ManifoldEM import find_conformational_coords
    find_conformational_coords.op()


def energy_landscape(_):
    from ManifoldEM import energy_landscape
    energy_landscape.op()


def compute_trajectory(_):
    from ManifoldEM import trajectory
    trajectory.op()


def relion_reconstruct(star_file: str, relion_command: str, output_path: str):
    import subprocess
    mrc_file = os.path.join(output_path, star_file.removesuffix('.star') + '.mrc')
    subprocess.run([relion_command, '--i', star_file, '--o', mrc_file], capture_output=True)


def mrcs2mrc(_):
    import shutil, glob, multiprocessing, tqdm
    from functools import partial

    relion_command = shutil.which('relion_reconstruct')
    if not relion_command:
        print("Can't find relion command 'relion_reconstruct'. Please verify that it is in your path")
        return

    print(f"Found relion command: '{relion_command}'")

    curr_path = os.getcwd()
    output_path = os.path.realpath(params.postproc_mrcs2mrc_dir)
    os.makedirs(output_path, exist_ok=True)
    os.chdir(params.bin_dir)

    reconstruct_local = partial(relion_reconstruct, relion_command=relion_command, output_path=output_path)
    star_files = glob.glob('*.star')
    if not star_files:
        print("No star files found for project. Have you run the 'trajectory' step?")
        return

    print(f"Converting {len(star_files)} star+mrcs files to mrc")
    with multiprocessing.Pool(processes=params.ncpu) as pool:
        for _ in tqdm.tqdm(enumerate(pool.imap_unordered(reconstruct_local, star_files)),
                              total=len(star_files)):
            pass

    print(f"Output in: {output_path}")

    os.chdir(curr_path)


def denoise_helper(i_bin: int, f: int, k: int, filter_type: str):
    import mrcfile
    import numpy as np
    from scipy import ndimage

    rec_file = os.path.join(params.postproc_mrcs2mrc_dir, 'EulerAngles_{}_{}_of_{}.mrc'.format(params.traj_name, i_bin + 1, params.states_per_coord))
    with mrcfile.open(rec_file) as mrc:
        vol = mrc.data
        vol = vol.astype(np.float64)
    if filter_type == 'gaussian':
        vol = ndimage.gaussian_filter(vol,k)
    elif filter_type == 'median':
        vol = ndimage.median_filter(vol, k)
    else:
        return

    rec1_file = os.path.join(params.postproc_denoise_dir, 'DenoiseimgsRELION_{}_{}_of_{}.mrc'.format(params.traj_name, i_bin + 1, params.states_per_coord))
    mrc = mrcfile.new(rec1_file)
    mrc.set_data(vol.astype(np.float32))


def denoise(args):
    from functools import partial
    import multiprocessing
    import tqdm
    f = args.frame
    k = args.window_size
    filter_type = args.filter.lower()

    bins = list(range(f)) + list(range(params.states_per_coord - f, params.states_per_coord))
    denoise_local = partial(denoise_helper, f=f, k=k, filter_type=filter_type)

    os.makedirs(params.postproc_denoise_dir, exist_ok=True)

    with multiprocessing.Pool(processes=params.ncpu) as pool:
        for _ in tqdm.tqdm(enumerate(pool.imap_unordered(denoise_local, bins)),
                           total=len(bins)):
            pass

    print(f"Output in: {os.path.realpath(params.postproc_denoise_dir)}")


_funcs = {
    "init": init,
    "threshold": threshold,
    "calc-distance": calc_distance,
    "manifold-analysis": manifold_analysis,
    "psi-analysis": psi_analysis,
    "nlsa-movie": nlsa_movie,
    "find-ccs": find_conformational_coordinates,
    "energy-landscape": energy_landscape,
    "trajectory": compute_trajectory,
    "mrcs2mrc": mrcs2mrc,
    "denoise": denoise,
}


def main():
    print(f"ManifoldEM version: {ManifoldEM.__version__}\n")
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    main_args = parser.parse_args()

    load_state(main_args)
    _funcs[main_args.command](main_args)


if __name__ == "__main__":
    main()
