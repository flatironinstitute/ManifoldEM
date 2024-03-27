import os
if not 'OMP_NUM_THREADS' in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Union
import ManifoldEM
from ManifoldEM.params import params, ProjectLevel
from . import interactive as mem

def get_parser():
    parser = ArgumentParser(
        prog="manifold-cli",
        description="Command-line interface for ManifoldEM package",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', '--ncpu', type=int, default=1)
    subparsers = parser.add_subparsers(help=None, dest="command")

    def add_relevant_params(subparser, level: ProjectLevel, prefix: str = ""):
        annotated_params = params.get_params_for_level(level)
        for param, (paramtype, paraminfo) in annotated_params.items():
            if paraminfo.user_param:
                default = getattr(params, param)
                subparser.add_argument(f"--{param}", metavar=paramtype.__name__.upper(), type=paramtype, default=default,
                                       help=f'{prefix}{paraminfo.description}')


    init_parser = subparsers.add_parser("init", help="0: Initialize new project", formatter_class=ArgumentDefaultsHelpFormatter)
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

    threshold_parser = subparsers.add_parser("threshold", help="1: Set upper/lower thresholds for principal direction detection",
                                             formatter_class=ArgumentDefaultsHelpFormatter)
    threshold_parser.add_argument("input_file", type=str)
    add_relevant_params(threshold_parser, ProjectLevel.BINNING)

    distance_parser = subparsers.add_parser("calc-distance", help="2: Calculate S2 distances",
                                            formatter_class=ArgumentDefaultsHelpFormatter)
    distance_parser.add_argument("input_file", type=str)
    distance_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(distance_parser, ProjectLevel.CALC_DISTANCE)

    manifold_analysis_parser = subparsers.add_parser("manifold-analysis", help="4: Initial embedding",
                                                     formatter_class=ArgumentDefaultsHelpFormatter)
    manifold_analysis_parser.add_argument("input_file", type=str)
    manifold_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(manifold_analysis_parser, ProjectLevel.MANIFOLD_ANALYSIS)

    psi_analysis_parser = subparsers.add_parser("psi-analysis", help="5: Analyze images to get psis",
                                                formatter_class=ArgumentDefaultsHelpFormatter)
    psi_analysis_parser.add_argument("input_file", type=str)
    psi_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(psi_analysis_parser, ProjectLevel.PSI_ANALYSIS)

    nlsa_movie_parser = subparsers.add_parser("nlsa-movie", help="6: Create 2D psi movies",
                                              formatter_class=ArgumentDefaultsHelpFormatter)
    nlsa_movie_parser.add_argument("input_file", type=str)
    nlsa_movie_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(nlsa_movie_parser, ProjectLevel.NLSA_MOVIE)

    cc_parser = subparsers.add_parser("find-ccs", help="7: Find conformational coordinates",
                                      formatter_class=ArgumentDefaultsHelpFormatter)
    cc_parser.add_argument("input_file", type=str)
    add_relevant_params(cc_parser, ProjectLevel.FIND_CCS)

    el_parser = subparsers.add_parser("energy-landscape", help="8: Calculate energy landscape",
                                      formatter_class=ArgumentDefaultsHelpFormatter)
    el_parser.add_argument("input_file", type=str)
    add_relevant_params(el_parser, ProjectLevel.ENERGY_LANDSCAPE)

    traj_parser = subparsers.add_parser("trajectory", help="9: Calculate trajectory",
                                        formatter_class=ArgumentDefaultsHelpFormatter)
    traj_parser.add_argument("input_file", type=str)
    add_relevant_params(traj_parser, ProjectLevel.TRAJECTORY)

    utility_parser = subparsers.add_parser("utility", help="Utility functions",
                                           formatter_class=ArgumentDefaultsHelpFormatter)
    utility_subparsers = utility_parser.add_subparsers(help=None, dest="command")

    mrcs2mrc_parser = utility_subparsers.add_parser("mrcs2mrc",
                                                    help="Convert output of trajectory step from mrcs to mrc [requires working relion install in PATH]",
                                                    formatter_class=ArgumentDefaultsHelpFormatter)
    mrcs2mrc_parser.add_argument("input_file", type=str)

    denoise_parser = utility_subparsers.add_parser("denoise", help="Denoise output of mrcs2mrc postprocessing step",
                                                   formatter_class=ArgumentDefaultsHelpFormatter)
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
    if hasattr(args, "path_width"):
        if args.path_width is None:
            return
        if args.path_width < 1 or args.path_width > 5:
            print("path-width argument must be on the interval [1, 5]")
            sys.exit(1)
        params.width_1D = args.path_width

    params.save()


def _parse_prd_list(prd_list: str) -> Union[list[int], None]:
    if prd_list:
        return [int(i) for i in prd_list.split(',')]

    return None


def args_to_dict(args: Namespace) -> dict:
    kwargs = vars(args)
    if 'prds' in kwargs.keys():
        kwargs['prd_list'] = _parse_prd_list(kwargs.pop('prds'))
    if 'ncpu' in kwargs.keys():
        params.ncpu = kwargs.pop('ncpu')
        params.save()

    return kwargs


def threshold(args):
    params.project_level = ProjectLevel.BINNING
    params.save()


def init(args: Namespace):
    mem.init(**args_to_dict(args))


def threshold(args: Namespace):
    mem.threshold(**args_to_dict(args))


def calc_distance(args: Namespace):
    mem.calc_distance(**args_to_dict(args))


def manifold_analysis(args: Namespace):
    mem.manifold_analysis(**args_to_dict(args))


def psi_analysis(args: Namespace):
    mem.psi_analysis(**args_to_dict(args))


def nlsa_movie(args: Namespace):
    mem.nlsa_movie(**args_to_dict(args))


def find_conformational_coordinates(args: Namespace):
    mem.find_conformational_coordinates(**args_to_dict(args))


def energy_landscape(args: Namespace):
    mem.energy_landscape(**args_to_dict(args))


def compute_trajectory(args: Namespace):
    mem.compute_trajectory(**args_to_dict(args))


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


def denoise_helper(i_bin: int, k: int, filter_type: str):
    import mrcfile
    import numpy as np
    from scipy import ndimage

    rec_file = os.path.join(params.postproc_mrcs2mrc_dir, 'EulerAngles_{}_{}_of_{}.mrc'.format(params.traj_name, i_bin + 1, params.states_per_coord))
    with mrcfile.open(rec_file) as mrc:
        vol = mrc.data
        vol = vol.astype(np.float64)
    if filter_type == 'gaussian':
        vol = ndimage.gaussian_filter(vol, k)
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
    denoise_local = partial(denoise_helper, k=k, filter_type=filter_type)

    os.makedirs(params.postproc_denoise_dir, exist_ok=True)

    with multiprocessing.Pool(processes=params.ncpu) as pool:
        for _ in tqdm.tqdm(enumerate(pool.imap_unordered(denoise_local, bins)),
                           total=len(bins)):
            pass

    print(f"Output in: {os.path.realpath(params.postproc_denoise_dir)}")


def set_params(args):
    for attr in dir(args):
        if attr.startswith('_') or not hasattr(params, attr):
            continue

        curr_value = getattr(params, attr)
        new_value = getattr(args, attr)
        if new_value != curr_value:
            print(f"Changing param {attr} from {curr_value} to {new_value}")
            setattr(params, attr, new_value)


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
    set_params(main_args)
    _funcs[main_args.command](main_args)


if __name__ == "__main__":
    main()
