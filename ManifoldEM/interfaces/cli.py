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


    init_parser = subparsers.add_parser("init", help="Initialize new project")
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

    threshold_parser = subparsers.add_parser("threshold", help="Set upper/lower thresholds for principal direction detection")
    threshold_parser.add_argument("input_file", type=str)
    add_relevant_params(threshold_parser, ProjectLevel.BINNING)

    distance_parser = subparsers.add_parser("calc-distance", help="Calculate S2 distances")
    distance_parser.add_argument("input_file", type=str)
    distance_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(distance_parser, ProjectLevel.CALC_DISTANCE)

    manifold_analysis_parser = subparsers.add_parser("manifold-analysis", help="Initial embedding")
    manifold_analysis_parser.add_argument("input_file", type=str)
    manifold_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(manifold_analysis_parser, ProjectLevel.MANIFOLD_ANALYSIS)

    psi_analysis_parser = subparsers.add_parser("psi-analysis", help="Analyze images to get psis")
    psi_analysis_parser.add_argument("input_file", type=str)
    psi_analysis_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(psi_analysis_parser, ProjectLevel.PSI_ANALYSIS)

    nlsa_movie_parser = subparsers.add_parser("nlsa-movie", help="Create 2D psi movies")
    nlsa_movie_parser.add_argument("input_file", type=str)
    nlsa_movie_parser.add_argument("--prds", type=str, metavar="INT,INT,...", help="Comma delineated list of prds you wish to calculate -- useful for debugging")
    add_relevant_params(nlsa_movie_parser, ProjectLevel.NLSA_MOVIE)

    cc_parser = subparsers.add_parser("find-ccs", help="Find conformational coordinates")
    cc_parser.add_argument("input_file", type=str)
    add_relevant_params(cc_parser, ProjectLevel.FIND_CCS)

    el_parser = subparsers.add_parser("energy-landscape", help="Calculate energy landscape")
    el_parser.add_argument("input_file", type=str)
    add_relevant_params(el_parser, ProjectLevel.ENERGY_LANDSCAPE)

    traj_parser = subparsers.add_parser("trajectory", help="Calculate trajectory")
    traj_parser.add_argument("input_file", type=str)
    add_relevant_params(traj_parser, ProjectLevel.TRAJECTORY)

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
