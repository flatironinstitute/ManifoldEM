import os
if not 'OMP_NUM_THREADS' in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

import sys
from argparse import ArgumentParser
import ManifoldEM

def get_parser():
    parser = ArgumentParser(
        prog="manifold-cli",
        description="Command-line interface for ManifoldEM package",
    )
    parser.add_argument('-n', '--ncpu', type=int, default=1)
    subparsers = parser.add_subparsers(help=None, dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument('-p', "--project-name", type=str, help="Name of project to create", required=True)
    init_parser.add_argument('-v', "--avg-volume", type=str, default="")
    init_parser.add_argument('-a', "--alignment", type=str, default="")
    init_parser.add_argument('-i', "--image-stack", type=str, default="")
    init_parser.add_argument('-m', "--mask-volume", type=str, default="")
    init_parser.add_argument('-s', "--pixel-size", type=float, required=True)
    init_parser.add_argument('-d', "--diameter", type=float, required=True)
    init_parser.add_argument('-r', "--resolution", type=float, required=True)
    init_parser.add_argument('-x', "--aperture-index", type=int, default=1)
    init_parser.add_argument('-o', '--overwrite', action='store_true',
                             help="Replace existing project with same name automatically")


    threshold_parser = subparsers.add_parser("threshold", help="Set upper/lower thresholds for principal direction detection")
    threshold_parser.add_argument("input_file", type=str)
    threshold_parser.add_argument("--low", type=int, default=100, help="Minimum number of images in a bin to count as a valid principle direction")
    threshold_parser.add_argument("--high", type=int, default=2000, help="Maximum number of images to analyze in a single principle direction")

    distance_parser = subparsers.add_parser("calc-distance", help="Calculate S2 distances")
    distance_parser.add_argument("input_file", type=str)
    distance_parser.add_argument("--num-psis", type=int, default=8, help="Number of eigenvectors to use for NLSA analysis")
    distance_parser.add_argument("--prds", type=str, help="Comma delineated list of prds you wish to calculate -- useful for debugging")

    manifold_analysis_parser = subparsers.add_parser("manifold-analysis", help="Initial embedding")
    manifold_analysis_parser.add_argument("input_file", type=str)
    manifold_analysis_parser.add_argument("--prds", type=str, help="Comma delineated list of prds you wish to calculate -- useful for debugging")

    psi_analysis_parser = subparsers.add_parser("psi-analysis", help="Analyze images to get psis")
    psi_analysis_parser.add_argument("input_file", type=str)
    psi_analysis_parser.add_argument("--prds", type=str, help="Comma delineated list of prds you wish to calculate -- useful for debugging")

    nlsa_movie_parser = subparsers.add_parser("nlsa-movie", help="Create 2D psi movies")
    nlsa_movie_parser.add_argument("input_file", type=str)
    nlsa_movie_parser.add_argument("--prds", type=str, help="Comma delineated list of prds you wish to calculate -- useful for debugging")

    cc_parser = subparsers.add_parser("find-ccs", help="Find conformational coordinates")
    cc_parser.add_argument("input_file", type=str)

    el_parser = subparsers.add_parser("energy-landscape", help="Calculate energy landscape")
    el_parser.add_argument("input_file", type=str)

    traj_parser = subparsers.add_parser("trajectory", help="Calculate trajectory")
    traj_parser.add_argument("input_file", type=str)
    traj_parser.add_argument("--path-width", type=int)

    return parser


def load_state(args):
    from ManifoldEM.params import p

    if args.command == "init":
        return
    fname_front = args.input_file.split('params_', 1)[1]
    fname_sans = os.path.splitext(fname_front)[0]
    p.proj_name = fname_sans

    p.load(args.input_file)

    p.ncpu = args.ncpu
    if hasattr(args, "num_psis"):
        p.num_psis = args.num_psis
    if args.command == "threshold":
        p.PDsizeThL = args.low
        p.PDsizeThH = args.high
    if hasattr(args, "path_width"):
        if args.path_width is None:
            return
        if args.path_width < 1 or args.path_width > 5:
            print("path-width argument must be on the interval [1, 5]")
            sys.exit(1)
        p.width_1D = args.path_width

    p.save()


def init(args):
    import shutil
    from ManifoldEM.params import p
    from ManifoldEM.util import get_image_width_from_stack

    p.proj_name = args.project_name
    proj_file = f'params_{p.proj_name}.toml'

    if os.path.isfile(proj_file) or os.path.isdir(p.out_dir):
        response = 'y' if args.overwrite else None
        while response not in ('y', 'n'):
            response = input("Project appears to exist. Overwrite? y/n\n").lower()
        if response == 'n':
            print("Aborting")
            return 1
        print("Removing previous project")
        if os.path.isdir(p.out_dir):
            shutil.rmtree(p.out_dir)

    p.avg_vol_file = os.path.expanduser(args.avg_volume)
    p.align_param_file = os.path.expanduser(args.alignment)
    p.img_stack_file = os.path.expanduser(args.image_stack)
    p.mask_vol_file = os.path.expanduser(args.mask_volume)

    p.pix_size = args.pixel_size
    p.obj_diam = args.diameter
    p.resol_est = args.resolution
    p.ap_index = args.aperture_index
    p.relion_data = args.alignment.endswith('.star')

    p.nPix = get_image_width_from_stack(p.img_stack_file)

    p.create_dir()
    p.save()


def _parse_prd_list(prd_list: str):
    if prd_list:
        return [int(i) for i in prd_list.split(',')]

    return None


def threshold(args):
    from ManifoldEM.params import p

    p.PDsizeThL = args.low
    p.PDsizeThH = args.high
    p.resProj = 2
    p.save()


def calc_distance(args):
    from ManifoldEM import GetDistancesS2
    prd_list = _parse_prd_list(args.prds)

    GetDistancesS2.op(prd_list)


def manifold_analysis(args):
    from ManifoldEM import manifoldAnalysis
    prd_list = _parse_prd_list(args.prds)

    manifoldAnalysis.op(prd_list)


def psi_analysis(args):
    from ManifoldEM import psiAnalysis
    prd_list = _parse_prd_list(args.prds)

    psiAnalysis.op(prd_list)


def nlsa_movie(args):
    from ManifoldEM import NLSAmovie
    from ManifoldEM.params import p

    prd_list = _parse_prd_list(args.prds)

    NLSAmovie.op(prd_list)
    if prd_list:
        p.resProj = 3
        p.save()


def find_conformational_coordinates(_):
    from ManifoldEM import FindConformationalCoord
    FindConformationalCoord.op()


def energy_landscape(_):
    from ManifoldEM import EL1D
    from ManifoldEM.params import p
    EL1D.op()
    p.resProj = 5
    p.save()


def compute_trajectory(_):
    from ManifoldEM import PrepareOutputS2
    PrepareOutputS2.op()


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
