from ManifoldEM.params import params, ProjectLevel
import os
import shutil
from typing import Union
from threadpoolctl import threadpool_limits

def init(project_name: str, avg_volume: str, alignment: str, image_stack: str, mask_volume: str,
         pixel_size: float, diameter: float, resolution: float, aperture_index: int, overwrite: bool,
         **kwargs):
    from ManifoldEM.util import get_image_width_from_stack
    import multiprocessing

    params.project_name = project_name
    proj_file = f'params_{params.project_name}.toml'

    if os.path.isfile(proj_file) or os.path.isdir(params.out_dir):
        response = 'y' if overwrite else None
        while response not in ('y', 'n'):
            response = input("Project appears to exist. Overwrite? y/n\n").lower()
        if response == 'n':
            print("Aborting")
            return 1
        print("Removing previous project")
        if os.path.isdir(params.out_dir):
            shutil.rmtree(params.out_dir)

    params.avg_vol_file = os.path.expanduser(avg_volume)
    params.align_param_file = os.path.expanduser(alignment)
    params.img_stack_file = os.path.expanduser(image_stack)
    params.mask_vol_file = os.path.expanduser(mask_volume)

    params.ms_pixel_size = pixel_size
    params.particle_diameter = diameter
    params.ms_estimated_resolution = resolution
    params.aperture_index = aperture_index
    params.is_relion_data = alignment.endswith('.star')

    params.ms_num_pixels = get_image_width_from_stack(params.img_stack_file)
    params.ncpu = multiprocessing.cpu_count()

    params.create_dir()
    params.save()


def load(project_file: str):
    params.load(project_file)


def threshold(**kwargs):
    params.project_level = ProjectLevel.BINNING
    params.save()


def calc_distance(prd_list: Union[list[int], None] = None, blas_threads=1, **kwargs):
    from ManifoldEM.calc_distance import op as calc_distance
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        calc_distance(prd_list)


def manifold_analysis(prd_list: Union[list[int], None] = None, blas_threads=1, **kwargs):
    from ManifoldEM.manifold_analysis import op as manifold_analysis
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        manifold_analysis(prd_list)


def psi_analysis(prd_list: Union[list[int], None] = None, blas_threads=1, **kwargs):
    from ManifoldEM.psi_analysis import op as psi_analysis
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        psi_analysis(prd_list)


def nlsa_movie(prd_list: Union[list[int], None] = None, blas_threads=1, **kwargs):
    from ManifoldEM.nlsa_movie import op as nlsa_movie
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        nlsa_movie(prd_list)


def find_conformational_coordinates(blas_threads=1, **kwargs):
    from ManifoldEM.find_conformational_coords import op as find_conformational_coords
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        find_conformational_coords()


def energy_landscape(blas_threads=1, **kwargs):
    from ManifoldEM.energy_landscape import op as energy_landscape
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        energy_landscape()


def compute_trajectory(blas_threads=1, **kwargs):
    from ManifoldEM.trajectory import op as trajectory
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        trajectory()
