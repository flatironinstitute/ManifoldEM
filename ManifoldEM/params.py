"""A place to hold parameters and generate paths.

This was implemented originally as basically a global namespace (implemented via globals on the
module). To maintain that structure, but provide better functionality (like class @properties
and sanity checking), we use Params() as a singleton class, named 'p' for legacy reasons.

Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Suvrajit Maji 2019
Copyright (c) Flatiron Institute Robert Blackwell 2023
"""
import numpy as np
import os
from pprint import pprint
import toml
import traceback
from typing import Annotated


# project_state structure:
#     0: default; new project
#     1: import tab/thresholding complete
#     2: calc-distance complete
#     3: manifold-analysis complete
#     4: psi-analysis complete
#     5: nlsa-movie complete
#     6: PrD selection complete
#     7: find-ccs complete
#     8: energy-landscape complete
#     9: trajectory complete
class Params():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Params, cls).__new__(cls)
        return cls.instance

    proj_name: Annotated[str, 'Name of the project'] = ''
    project_state: Annotated[int, 'Current analysis level of project'] = 0
    relion_data: Annotated[bool, 'Is the code analyzing relion data?'] = True
    ncpu: Annotated[int, 'Number of processes to use for multiprocessing'] = 1
    avg_vol_file: Annotated[str, 'Average volume file (e.g., .mrc)'] = ''
    img_stack_file: Annotated[str, 'Image stack file (e.g., .mrcs)'] = ''
    align_param_file: Annotated[str, 'Alignment file (e.g., .star)'] = ''
    mask_vol_file: Annotated[str, 'Mask volume file (e.g., .mrc)'] = ''
    num_part: Annotated[int, 'Total number of particles in stack'] = 0
    Cs: Annotated[float, 'Spherical Aberration [mm] (from alignment file)'] = 0.0
    EkV: Annotated[float, 'Voltage [kV] (from alignment file)'] = 0.0
    AmpContrast: Annotated[float, 'Amplitude Contrast [ratio](from alignment file)'] = 0.0
    eps: Annotated[float, 'Small fraction to be added if divide-by-zero errors occur'] = 1E-10
    gaussEnv: Annotated[float, 'envelope of CTF'] = float("inf")

    # microscopy parameters:
    nPix: Annotated[int, 'Window size of image (e.g., for 100x100 image, nPix=100)'] = 0
    pix_size: Annotated[float, 'Pixel size of image [Angstroms] (known via rln_DetectorPixelSize*10e6 / rln_Magnification)'] = 0.0
    obj_diam: Annotated[float, 'Diameter of macromolecule [Angstroms]'] = 0.0
    resol_est: Annotated[float, 'Estimated resolution [Angstroms]'] = 0.0
    ap_index: Annotated[int, 'Aperture index {1,2,3...}; increases tessellated bin size'] = 1

    # tessellation binning:
    PDsizeThL: Annotated[int, 'Minimum required snapshots in a tessellation for it be admitted'] = 100
    PDsizeThH: Annotated[int, 'Maximum number of snapshots that will be considered within each tessellation'] = 2000
    numberofJobs: Annotated[int, 'Total number of bins to consider for manifold embedding'] = 0

    # eigenfunction parameters:
    num_eigs: Annotated[int, 'Number of highest-eigenvalue eigenfunctions to consider in total (max entry of eigenvalue spectrum)'] = 15
    num_psiTrunc: Annotated[int, 'Number of eigenfunctions for truncated views'] = 8
    num_psis: Annotated[int, 'Number of eigenfunctions'] = 8
    rad: Annotated[int, 'Manifold pruning'] = 5

    # NLSA parameters:
    fps: Annotated[float, 'Frames per second for generated movies'] = 5.0
    conOrderRange: Annotated[int, 'Coarse-graining factor of energy landscape'] = 50
    # tune automation suggestion (Ali): larger tune = smaller gaussian width; turns data into
    # islands/chunks (can't see long-range patterns); a good tune should form a 'good' psirec parabola.
    # as well, you can keep using tune to get rid of outliers in data; you want the number of outliers
    # to be around 10%; if higher than this, tune needs to be changed.
    tune: Annotated[int, 'Diffusion map tuning parameter'] = 3

    # energy landscape parameters:
    dim: Annotated[int, 'Number of reaction coordinates for energy landscape'] = 1
    temperature: Annotated[int, 'User-defined pre-quenching temperature of experiments'] = 25
    trajName: Annotated[str, 'Filename modifier for exported (2D) trajectories'] = '1'
    nClass: Annotated[int, 'Number of states partitioned within each 1D reaction coordinate; results in a nClassx1 1D ELS'] = 50
    width_1D: Annotated[int, 'Width of trajectory in 1D energy path'] = 1
    width_2D: Annotated[int, 'Width of trajectory in 2D energy path'] = 1

    #  reaction coordinates parameters:
    getOpticalFlow: Annotated[bool, 'Compute optical flow vectors'] = True
    getAllEdgeMeasures: Annotated[bool, 'Compute "edge measures"'] = True
    opt_movie: dict = {'printFig': 0,
                       'OFvisual': 0,
                       'visual_CC': 0,
                       'flowVecPctThresh': 95}
    opt_mask_type: Annotated[int, '0:None, 1:Annular, 2:Volumetric'] = 0
    opt_mask_param: Annotated[int, 'for either none, radius (Int), or iso(Int)'] = 0

    findBadPsiTau: Annotated[bool, ''] = True
    tau_occ_thresh: Annotated[float, ''] = 0.35
    use_pruned_graph: Annotated[bool, ''] = False

    visualization_params: dict = {
        'S2_scale': 1.0,
        'S2_density': 1000,
        'S2_isosurface_level': 3,
    }

    @property
    def proj_file(self) -> str:
        return f'params_{self.proj_name}.toml'

    @property
    def ang_width(self) -> float:
        if not self.obj_diam:
            return 0.0
        return np.min(((self.ap_index * self.resol_est) / self.obj_diam, np.sqrt(4 * np.pi)))

    @property
    def sh(self) -> float:
        return self.resol_est / self.obj_diam


    @property
    def user_dir(self) -> str:
        return 'output'


    @property
    def out_dir(self) -> str:
        return os.path.join(self.user_dir, self.proj_name)


    @property
    def psi_file(self) -> str:
        return os.path.join(self.psi_dir, 'gC_trimmed_psi_')


    @property
    def psi2_file(self) -> str:
        return os.path.join(self.psi2_dir, 'S2_')


    @property
    def OM1_file(self) -> str:
        return os.path.join(self.OM_dir, 'S2_')


    @property
    def rho_file(self) -> str:
        return os.path.join(self.OM_dir, 'rho')


    @property
    def pd_file(self) -> str:
        return os.path.join(self.out_dir, 'pd_data')


    @property
    def dist_dir(self) -> str:
        return os.path.join(self.out_dir, 'distances')


    @property
    def dist_file(self) -> str:
        return os.path.join(self.dist_dir, 'IMGs_')


    @property
    def psi_dir(self) -> str:
        return os.path.join(self.out_dir, 'diff_maps')


    @property
    def psi2_dir(self) -> str:
        return os.path.join(self.out_dir, 'psi_analysis')


    @property
    def EL_dir(self) -> str:
        return os.path.join(self.out_dir, f'ELConc{self.conOrderRange}')


    @property
    def OM_dir(self) -> str:
        return os.path.join(self.EL_dir, 'OM')


    @property
    def EL_file(self) -> str:
        return os.path.join(self.EL_dir, 'S2_')


    @property
    def OM_file(self) -> str:
        return os.path.join(self.OM_dir, 'S2_')


    @property
    def traj_dir(self) -> str:
        return os.path.join(self.out_dir, 'traj')


    @property
    def CC_dir(self) -> str:
        return os.path.join(self.out_dir, 'CC')


    @property
    def CC_file(self) -> str:
        return os.path.join(self.CC_dir,  'CC_file')


    @property
    def CC_meas_dir(self) -> str:
        return os.path.join(self.CC_dir, 'CC_meas')


    @property
    def CC_meas_file(self) -> str:
        return os.path.join(self.CC_meas_dir, 'meas_edge_prDs_')


    @property
    def CC_OF_dir(self) -> str:
        return os.path.join(self.CC_dir, 'CC_OF')


    @property
    def CC_OF_file(self) -> str:
        return os.path.join(self.CC_OF_dir, 'OF_prD_')


    @property
    def traj_file(self) -> str:
        return os.path.join(self.traj_dir, 'traj_')


    @property
    def euler_dir(self) -> str:
        return os.path.join(self.out_dir, 'topos', 'Euler_PrD')


    @property
    def ref_ang_file(self) -> str:
        return os.path.join(self.euler_dir, 'PrD_map.txt')


    @property
    def ref_ang_file1(self) -> str:
        return os.path.join(self.euler_dir, 'PrD_map1.txt')


    @property
    def bin_dir(self) -> str:
        return os.path.join(self.out_dir, 'bin')


    @property
    def user_dimensions(self) -> int:
        "Placeholder parameter for when we support 2d later"
        return 1


    def get_topos_path(self, prd: int, index: int) -> str:
        return os.path.join(self.out_dir, 'topos', f'PrD_{prd}', f'topos_{index}.png')


    def get_psi_gif(self, prd: int, index: int) -> str:
        return os.path.join(self.out_dir, 'topos', f'PrD_{prd}', f'psi_{index}.gif')


    def get_EL_file(self, prd_index: int):
        return f'{self.EL_file}prD_{prd_index}'


    def get_psi_file(self, prd_index: int):
        return f'{self.psi_file}prD_{prd_index}'


    def get_psi2_file(self, prd_index: int):
        return f'{self.psi2_file}prD_{prd_index}'


    def get_dist_file(self, prd_index: int):
        return f'{self.dist_file}prD_{prd_index}'


    def asdict(self):
        res = {}
        for var in dir(Params):
            objtype = getattr(Params, var)
            if not var.startswith('_') and isinstance(objtype, (int, list, dict, float, str)):
                res[var] = getattr(self, var)

        return res


    def save(self, outfile: str = ''):
        if outfile == '':
            outfile = f'params_{self.proj_name}.toml'
        res = {'params': self.asdict()}
        with open(outfile, 'w') as f:
            toml.dump(res, f)
            f.flush()


    def load(self, infile: str = ''):
        if infile == '':
            infile = f'params_{self.proj_name}.toml'

        with open(infile, 'r') as f:
            indict = toml.load(f)
            basepath = os.path.dirname(infile)
            if basepath:
                os.chdir(basepath)

        valid_params = dir(Params)
        for param, val in indict['params'].items():
            if param not in valid_params:
                print(traceback.format_stack()[-2].split('\n')[0])
                print(f"Warning: param '{param}' not found in parameters module")
            else:
                setattr(self, param, val)


    def print(self):
        pprint(self.asdict())


    def create_dir(self):
        os.makedirs(self.dist_dir, exist_ok=True)
        os.makedirs(self.psi_dir, exist_ok=True)
        os.makedirs(self.psi2_dir, exist_ok=True)
        os.makedirs(self.EL_dir, exist_ok=True)
        os.makedirs(self.OM_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.CC_dir, exist_ok=True)
        os.makedirs(self.CC_OF_dir, exist_ok=True)
        os.makedirs(self.CC_meas_dir, exist_ok=True)
        os.makedirs(self.euler_dir, exist_ok=True)


p = Params()
