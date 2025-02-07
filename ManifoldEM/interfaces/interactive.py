from ManifoldEM.params import params, ProjectLevel
import os
import shutil
from typing import Union
from threadpoolctl import threadpool_limits

def init(project_name: str, avg_volume: str, alignment: str, image_stack: str, mask_volume: str,
         pixel_size: float, diameter: float, resolution: float, aperture_index: int, overwrite: bool,
         **kwargs):
    """
    Initializes a new project with specified parameters, including project directory creation
    and parameter storage.

    Parameters
    ----------
    project_name : str
        The name of the project.
    avg_volume : str
        Path to the file containing the average volume data.
    alignment : str
        Path to the file containing alignment parameters.
    image_stack : str
        Path to the file containing the image stack.
    mask_volume : str
        Path to the file containing the volume mask.
    pixel_size : float
        The pixel size of each image in Å.
    diameter : float
        The diameter of the object in Å.
    resolution : float
        The resolution estimate for the images in Å.
    aperture_index : int
        The index of the aperture used. Larger apertures increase area associated with a PrD.
    overwrite : bool
        Whether to overwrite an existing project with the same name.

    Notes
    -----
    - The function sets project parameters in a global variable `params`, which is an
      instance of a class with attributes corresponding to the project parameters and methods
      for directory creation (`create_dir`) and parameter storage (`save`).
    - It checks if the project directory already exists and handles it based on the `overwrite`
      option. If `overwrite` is True, the existing directory is removed; otherwise, the function
      exits without creating a new project.
    - The function determines whether the alignment file is in RELION's STAR format based on its
      file extension and sets a flag accordingly.
    - The image width is determined from the image stack file using the `get_image_width_from_stack`
      function.
    - Project parameters are saved to a TOML file using the `save` method of the global variable `params`.
    """

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


def probability_landscape(blas_threads=1, **kwargs):
    from ManifoldEM.probability_landscape import op as probability_landscape
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        probability_landscape()


def compute_trajectory(blas_threads=1, **kwargs):
    from ManifoldEM.trajectory import op as trajectory
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        trajectory()
