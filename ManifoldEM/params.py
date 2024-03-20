"""A place to hold parameters and generate paths.

This was implemented originally as basically a global namespace (implemented via globals on the
module). To maintain that structure, but provide better functionality (like class @properties
and sanity checking), we use Params() as a singleton class, named 'p' for legacy reasons.

Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Suvrajit Maji 2019
Copyright (c) Flatiron Institute Robert Blackwell 2023
"""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import os
from pprint import pprint
import toml
import traceback
from typing import Annotated, Dict, List, get_type_hints, get_args


class ProjectLevel(Enum):
    INIT = 0
    BINNING = 1
    CALC_DISTANCE = 2
    MANIFOLD_ANALYSIS = 3
    PSI_ANALYSIS = 4
    NLSA_MOVIE = 5
    PRD_SELECTION = 6
    FIND_CCS = 7
    ENERGY_LANDSCAPE = 8
    TRAJECTORY = 9


@dataclass
class ParamInfo():
    description: str = ''
    user_param: bool = False
    affects: List[ProjectLevel] = field(default_factory=list)


class Params():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Params, cls).__new__(cls)
        return cls.instance

    proj_name: Annotated[str, ParamInfo('Name of the project', False, [ProjectLevel.INIT])] = ''
    project_level: Annotated[ProjectLevel, ParamInfo('Current analysis level of project')] = ProjectLevel.INIT
    relion_data: Annotated[bool, ParamInfo('Is the code analyzing relion data?')] = True
    ncpu: Annotated[int, ParamInfo('Number of processes to use for multiprocessing', True)] = 1
    avg_vol_file: Annotated[str, ParamInfo('Average volume file (e.g., .mrc)')] = ''
    img_stack_file: Annotated[str, ParamInfo('Image stack file (e.g., .mrcs)')] = ''
    align_param_file: Annotated[str, ParamInfo('Alignment file (e.g., .star)')] = ''
    mask_vol_file: Annotated[str, ParamInfo('Mask volume file (e.g., .mrc)')] = ''
    num_part: Annotated[int, ParamInfo('Total number of particles in stack')] = 0
    Cs: Annotated[float, ParamInfo('Spherical Aberration [mm] (from alignment file)')] = 0.0
    EkV: Annotated[float, ParamInfo('Voltage [kV] (from alignment file)')] = 0.0
    AmpContrast: Annotated[float, ParamInfo('Amplitude Contrast [ratio](from alignment file)')] = 0.0
    eps: Annotated[float, ParamInfo('Small fraction to be added if divide-by-zero errors occur', True,
                                    [ProjectLevel.BINNING])] = 1E-10
    gaussEnv: Annotated[float, ParamInfo('Envelope of CTF')] = float("inf")

    # microscopy parameters:
    nPix: Annotated[int, ParamInfo('Window size of image (e.g., for 100x100 image, nPix=100)')] = 0
    pix_size: Annotated[float, ParamInfo('Pixel size of image [Angstroms] (known via rln_DetectorPixelSize*10e6 / rln_Magnification)')] = 0.0
    obj_diam: Annotated[float, ParamInfo('Diameter of macromolecule [Angstroms]')] = 0.0
    resol_est: Annotated[float, ParamInfo('Estimated resolution [Angstroms]')] = 0.0
    ap_index: Annotated[int, ParamInfo('Aperture index {1,2,3...}; increases tessellated bin size')] = 1

    # tessellation binning:
    PDsizeThL: Annotated[int, ParamInfo('Minimum required snapshots in a tessellation for it be admitted', True,
                                        [ProjectLevel.BINNING])] = 100
    PDsizeThH: Annotated[int, ParamInfo('Maximum number of snapshots that will be considered within each tessellation', True,
                                        [ProjectLevel.BINNING])] = 2000
    numberofJobs: Annotated[int, ParamInfo('Total number of bins to consider for manifold embedding')] = 0

    # Distance calculation parameters
    distance_filter_type: Annotated[str, ParamInfo('Filter type for image preprocessing. Valid: {"Butter", "Gauss"}', True,
                                                   [ProjectLevel.CALC_DISTANCE])] = 'Butter'
    distance_filter_cutoff_freq: Annotated[float, ParamInfo('Nyquist cutoff frequency for filter', True, [ProjectLevel.CALC_DISTANCE])] = 0.5
    distance_filter_order: Annotated[int, ParamInfo('Order of Filter ("Butter" only)', True, [ProjectLevel.CALC_DISTANCE])] = 8

    # eigenfunction parameters:
    num_eigs: Annotated[int, ParamInfo('Number of highest-eigenvalue eigenfunctions to consider in total (max entry of eigenvalue spectrum)')] = 15
    num_psiTrunc: Annotated[int, ParamInfo('Number of eigenfunctions for truncated views')] = 8
    num_psis: Annotated[int, ParamInfo('Number of eigenfunctions for analysis', True, [ProjectLevel.BINNING])] = 8
    rad: Annotated[int, ParamInfo('Manifold pruning'), True, [ProjectLevel.MANIFOLD_ANALYSIS]] = 5

    # NLSA parameters:
    fps: Annotated[float, ParamInfo('Frames per second for generated movies', True, [ProjectLevel.NLSA_MOVIE])] = 5.0
    conOrderRange: Annotated[int, ParamInfo('Coarse-graining factor of energy landscape', True,
                                            [ProjectLevel.PSI_ANALYSIS,
                                             ProjectLevel.ENERGY_LANDSCAPE,
                                             ProjectLevel.TRAJECTORY])] = 50
    # tune automation suggestion (Ali): larger tune = smaller gaussian width; turns data into
    # islands/chunks (can't see long-range patterns); a good tune should form a 'good' psirec parabola.
    # as well, you can keep using tune to get rid of outliers in data; you want the number of outliers
    # to be around 10%; if higher than this, tune needs to be changed.
    tune: Annotated[int, ParamInfo('Diffusion map tuning parameter', True,
                                   [ProjectLevel.MANIFOLD_ANALYSIS, ProjectLevel.PSI_ANALYSIS])] = 3

    # energy landscape parameters:
    dim: Annotated[int, ParamInfo('Number of reaction coordinates for energy landscape')] = 1
    temperature: Annotated[int, ParamInfo('User-defined pre-quenching temperature of experiments')] = 25
    trajName: Annotated[str, ParamInfo('Filename modifier for exported (2D) trajectories')] = '1'
    nClass: Annotated[int, ParamInfo('Number of states partitioned within each 1D reaction coordinate; results in a nClassx1 1D ELS')] = 50
    width_1D: Annotated[int, ParamInfo('Width of trajectory in 1D energy path')] = 1
    width_2D: Annotated[int, ParamInfo('Width of trajectory in 2D energy path')] = 1

    #  reaction coordinates parameters:
    getOpticalFlow: Annotated[bool, ParamInfo('Compute optical flow vectors')] = True
    getAllEdgeMeasures: Annotated[bool, ParamInfo('Compute "edge measures"')] = True
    opt_movie: dict = {'printFig': 0,
                       'OFvisual': 0,
                       'visual_CC': 0,
                       'flowVecPctThresh': 95}
    opt_mask_type: Annotated[int, ParamInfo('0:None, 1:Annular, 2:Volumetric')] = 0
    opt_mask_param: Annotated[int, ParamInfo('for either none, radius (Int), or iso(Int)')] = 0

    findBadPsiTau: Annotated[bool, ParamInfo('')] = True
    tau_occ_thresh: Annotated[float, ParamInfo('')] = 0.35
    use_pruned_graph: Annotated[bool, ParamInfo('')] = False

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


    def get_user_params(self) -> Dict[str, Annotated]:
        return {k: v for k, v in get_type_hints(self, include_extras=True).items()
                if hasattr(v, '__metadata__') and v.__metadata__[0].user_param }


    def get_params_for_level(self, level: ProjectLevel, first_appearance=True):
        res = dict()
        for k, v in get_type_hints(self, include_extras=True).items():
            if not hasattr(v, '__metadata__') or level not in v.__metadata__[0].affects:
                continue

            if not first_appearance or level.value == min([a.value for a in v.__metadata__[0].affects], default=-1):
                res[k] = (get_args(v)[0], v.__metadata__[0],)

        return res


    def asdict(self):
        res = {}
        for var in dir(Params):
            objtype = getattr(Params, var)
            if var.startswith('_'):
                continue

            val = getattr(self, var)
            if isinstance(objtype, (int, list, dict, float, str)):
                res[var] = val
            elif isinstance(objtype, ProjectLevel):
                res[var] = val.value

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
                if (param == 'project_level'):
                    setattr(self, param, ProjectLevel(val))
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
