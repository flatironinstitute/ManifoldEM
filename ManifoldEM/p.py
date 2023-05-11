"""A place to hold parameters and generate paths.

This was implemented originally as basically a global namespace (implemented via globals on the
module). To maintain that structure, but provide better functionality (like class @properties
and sanity checking), we use a weird scheme where the module itself is a class instance of
Params.

Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Suvrajit Maji 2019
Copyright (c) Flatiron Institute Robert Blackwell 2023

"""
import os
import sys
import toml
from pprint import pprint

import numpy as np

from ManifoldEM.util import debug_print

# resProj structure:
#     0: default; new project
#     1: user has confirmed Data.py entries
#     2: GetDistances.py complete
#     3: manifoldAnalysis.py complete
#     4: psiAnalysis.py complete
#     5: NLSAmovie.py complete
#     6: PD anchors chosen/saved, "Compile" button clicked
#     7: FindReactionCoord.py complete
#     8: EL1D.py complete
#     9: PrepareOutputS2.py complete
class Params(sys.__class__):
    proj_name: str = ''            # name of the project :D
    resProj: int = 0               # see above
    relion_data: bool = True       # working with relion data?
    ncpu: int = 1                  # number of CPUs for multiprocessing
    avg_vol_file: str = ''         # average volume file (e.g., .mrc)
    img_stack_file: str = ''       # image stack file (e.g., .mrcs)
    align_param_file: str = ''     # alignment file (e.g., .star)
    mask_vol_file: str = ''        # mask volume file (e.g., .mrc)
    num_part: int = 0              # total number of particles in stack
    Cs: float = 0.0                # Spherical Aberration [mm] (from alignment file)
    EkV: float = 0.0               # Voltage [kV] (from alignment file)
    AmpContrast: float = 0.0       # Amplitude Contrast [ratio](from alignment file);
    eps: float = 1E-10             # small fraction to be added if divide-by-zero errors occur
    gaussEnv: float = float("inf") # envelope of CTF

    # microscopy parameters:
    nPix: int = 0          # window size of image (e.g., for 100x100 image, nPix=100)
    pix_size: float = 0.0  # pixel size of image [Angstroms] (known via rln_DetectorPixelSize*10e6 / rln_Magnification)
    obj_diam: float = 0.0  # diameter of macromolecule [Angstroms]
    resol_est: float = 0.0 # estimated resolution [Angstroms]
    ap_index: int = 1      # aperture index {1,2,3...}; increases tessellated bin size
    sh: float = 0.0        # Shannon angle (pix_size / obj_diam)

    # tessellation binning:
    PDsizeThL: int = 100   # minimum required snapshots in a tessellation for it be admitted
    PDsizeThH: int = 2000  # maximum number of snapshots that will be considered within each tessellation
    S2rescale: float = 1.0 # proper scale ratio between S2 sphere and .mrc volume for visualizations
    S2iso: int = 3         # proper isosurface level of .mrv volume for vizualiaztion (as chosen by user)
    numberofJobs: int = 0  # total number of bins to consider for manifold embedding


    # eigenfunction parameters:
    num_eigs: int = 15      # number of highest-eigenvalue eigenfunctions to consider in total (max entry of eigenvalue spectrum)
    num_psiTrunc: int = 8    # number of eigenfunctions for truncated views
    num_psis: int = 8
    # tune automation suggestion (Ali): larger tune = smaller gaussian width; turns data into
    # islands/chunks (can't see long-range patterns); a good tune should form a 'good' psirec parabola.
    # as well, you can keep using tune to get rid of outliers in data; you want the number of outliers
    # to be around 10%; if higher than this, tune needs to be changed.
    tune: int = 3           # diffusion map tuning; this needs to be automated
    rad: int = 5            # manifold pruning
    conOrderRange: int = 50 # coarse-graining factor of energy landscape

    # NLSA movie parameters:
    fps: float = 5.0

    # energy landscape parameters:
    dim: int = 1              # user-defined number of dimensions (reaction coordinates); {1,2}
    temperature: int = 25     # user-defined pre-quenching temperature of experiments
    trajName: str = '1'       # filename for exported (2D) trajectories
    nClass: int = 50          # number of states partitioned within each 1D reaction coordinate; results in a 50x1 1D ELS
    width_1D: int = 1         # user-defined width of trajectory in 1D energy path
    width_2D: int = 1         # user-defined width of trajectory in 2D energy path
    hUh = None                # occupancy map


    #  reaction coordinates parameters:
    getOpticalFlow: bool = True                # default True to compute optical flow vectors
    getAllEdgeMeasures: bool = True            # default True to compute edge measures
    anch_list: list[list[int]] = []            # user-defined PD anchors for Belief Propagation
    trash_list: list[bool] = []                # user-defined PD removals to ignore via final compile [binary list, 1 entry/PD]
    opt_movie: dict = {'printFig': 0,
                       'OFvisual': 0,
                       'visual_CC': 0,
                       'flowVecPctThresh': 95}
    opt_mask_type: int = 0                     # 0:None, 1:Annular, 2:Volumetric
    opt_mask_param: int = 0                    # for either none, radius (Int), or iso(Int)

    findBadPsiTau: bool = True
    tau_occ_thresh: float = 0.35
    use_pruned_graph: bool = False

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
        return os.path.join(self.user_dir, f'outputs_{self.proj_name}')


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
    def tess_file(self) -> str:
        return os.path.join(self.out_dir, 'selecGCs')


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
    def CC_graph_file(self) -> str:
        return os.path.join(self.CC_dir, 'graphCC')


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


    def get_EL_file(self, prd_index: int):
        return f'{self.EL_file}prD_{prd_index}'


    def get_psi_file(self, prd_index: int):
        return f'{self.psi_file}prD_{prd_index}'


    def get_psi2_file(self, prd_index: int):
        return f'{self.psi2_file}prD_{prd_index}'


    def get_dist_file(self, prd_index: int):
        return f'{self.dist_file}prD_{prd_index}'


    def set_trash_list(self, trash_list):
        setattr(self, 'trash_list', [bool(a) for a in trash_list])


    def get_trash_list(self):
        return np.array(self.trash_list, dtype=bool)


    def todict(self):
        res = {}
        for var in dir(Params):
            if not var.startswith('_') and isinstance(getattr(Params, var), (int, list, dict, float, str)):
                res[var] = getattr(self, var)

        return res


    def save(self, outfile: str = ''):
        if outfile == '':
            outfile = f'params_{self.proj_name}.toml'
        res = {'params': self.todict()}
        with open(outfile, 'w') as f:
            toml.dump(res, f)
            f.flush()


    def load(self, infile: str = ''):
        if infile == '':
            infile = f'params_{self.proj_name}.toml'

        with open(infile, 'r') as f:
            indict = toml.load(f)

        valid_params = dir(Params)
        for param, val in indict['params'].items():
            if param not in valid_params:
                debug_print(f"Warning: param '{param}' not found in parameters module")
            else:
                setattr(self, param, val)


    def print(self):
        pprint(self.todict())


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



sys.modules[__name__].__class__ = Params
