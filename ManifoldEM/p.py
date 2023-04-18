import numpy as np
import os
import sys
import toml
from pprint import pprint

from ManifoldEM.util import debug_print


'''
Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Suvrajit Maji 2019
'''
"""
resProj structure:
    0: default; new project
    1: user has confirmed Data.py entries, but not yet started (or partially started) GetDistancesS2.py
    2: GetDistances.py complete, possible partial-progress on manifoldAnalysis.py
    3: manifoldAnalysis.py complete, possible partial-progress on psiAnalysis.py
    4: psiAnalysis.py complete, possible partial-progress on NLSAmovie.py
    5: NLSAmovie.py complete
    6: PD anchors chosen/saved, "Compile" button clicked, possible partial-progress on FindReactionCoord.py
    7: FindReactionCoord.py complete, possible partial-progress on EL1D.py
    8: EL1D.py complete, possible partial-progress on PrepareOutputS2.py
    9: PrepareOutputS2.py complete
"""
class Params(sys.__class__):
    proj_name: str = ''            # name of the project :D
    resProj: int = 0               # see above
    relion_data: bool = False      # working with relion data?
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
    ang_width: float = 0.0 # angle width (via: ap_index * resol_est / obj_diam)
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
    anch_list: list[int] = []                  # user-defined PD anchors for Belief Propagation
    trash_list: list[bool] = []                # user-defined PD removals to ignore via final compile [binary list, 1 entry/PD]
    opt_movie: dict = {'printFig': 0,
                       'OFvisual': 0,
                       'visual_CC': 0,
                       'flowVecPctThresh': 95}
    opt_mask_type: int = 0                     # 0:None, 1:Annular, 2:Volumetric
    opt_mask_param: int = 0                    # for either none, radius (Int), or iso(Int)

    findBadPsiTau: bool = True
    tau_occ_thresh: float = 0.5
    use_pruned_graph: bool = False


    # FIXME: These really shouldn't be cached, or at the very least should be put in their own subdict...
    user_dir: str = ''  # Root directory for project
    tau_dir: str = ''
    OM_dir: str = ''
    Var_dir: str = ''
    NLSA_dir: str = ''
    traj_dir: str = ''
    bin_dir: str = ''
    relion_dir: str = ''
    CC_dir: str = ''
    CC_OF_dir: str = ''
    CC_meas_dir: str = ''
    out_dir: str = ''
    post_dir: str = ''
    vol_dir: str = ''
    svd_dir: str = ''
    anim_dir: str = ''
    dist_file: str = ''
    psi_file: str = ''
    psi2_file: str = ''
    movie2d_file: str = ''
    tau_file: str = ''
    OM1_file: str = ''
    Var_file: str = ''
    rho_file: str = ''
    remote_file: str = ''
    NLSA_file: str = ''
    traj_file: str = ''
    CC_file: str = ''
    CC_OF_file: str = ''
    CC_meas_file: str = ''
    CC_graph_file: str = ''
    ref_ang_file: str = ''
    ref_ang_file1: str = ''


    @property
    def tess_file(self) -> str:
        return os.path.join(self.out_dir, 'selecGCs')


    @property
    def dist_dir(self) -> str:
        return os.path.join(self.out_dir, 'distances')


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
        for var in dir(self):
            if not var.startswith('_') and isinstance(getattr(self, var), (int, list, dict, float, str)):
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
        # input and output directories and files

        self.out_dir = os.path.join(self.user_dir, f'outputs_{self.proj_name}')

        p = self
        os.makedirs(p.dist_dir, exist_ok=True)
        os.makedirs(p.psi_dir, exist_ok=True)
        os.makedirs(p.psi2_dir, exist_ok=True)
        os.makedirs(p.EL_dir, exist_ok=True)
        os.makedirs(p.OM_dir, exist_ok=True)

        p.Var_dir = os.path.join(p.user_dir, 'outputs_{}/Var/'.format(p.proj_name))
        os.makedirs(p.Var_dir, exist_ok=True)
        p.traj_dir = os.path.join(p.user_dir, 'outputs_{}/traj/'.format(p.proj_name))
        os.makedirs(p.traj_dir, exist_ok=True)

        p.relion_dir = bin_dir = os.path.join(p.user_dir, 'outputs_{}/bin/'.format(p.proj_name))
        os.makedirs(bin_dir, exist_ok=True)

        p.CC_dir = os.path.join(p.user_dir, 'outputs_{}/CC/'.format(p.proj_name))
        p.CC_OF_dir = os.path.join(p.CC_dir, 'CC_OF')
        os.makedirs(p.CC_OF_dir, exist_ok=True)

        p.CC_meas_dir = os.path.join(p.CC_dir, 'CC_meas')
        os.makedirs(p.CC_meas_dir, exist_ok=True)

        #################
        # post-processing:
        p.post_dir = os.path.join(p.user_dir, 'outputs_{}/post/'.format(proj_name))
        p.vol_dir = os.path.join(p.post_dir, '1_vol')
        p.svd_dir = os.path.join(p.post_dir, '2_svd')
        p.anim_dir = os.path.join(p.post_dir, '3_anim')
        os.makedirs(p.post_dir, exist_ok=True)
        os.makedirs(p.vol_dir, exist_ok=True)
        os.makedirs(p.svd_dir, exist_ok=True)
        os.makedirs(p.anim_dir, exist_ok=True)

        #################
        os.makedirs(os.path.join(p.out_dir, 'topos', 'Euler_PrD'), exist_ok=True)

        p.dist_file = '{}/IMGs_'.format(p.dist_dir)
        p.psi_file = '{}/gC_trimmed_psi_'.format(p.psi_dir)
        p.psi2_file = '{}/S2_'.format(p.psi2_dir)
        p.OM1_file = '{}/S2_'.format(p.OM_dir)
        p.Var_file = '{}/S2_'.format(p.Var_dir)
        p.rho_file = '{}/rho'.format(p.OM_dir)
        p.remote_file = '{}/rem_'.format(p.Var_dir)
        p.traj_file = '{}/traj_'.format(p.traj_dir)
        p.CC_graph_file = '{}graphCC'.format(p.CC_dir)
        p.CC_OF_file = '{}OF_prD_'.format(p.CC_OF_dir)
        p.CC_meas_file = '{}meas_edge_prDs_'.format(p.CC_meas_dir)
        p.CC_file = '{}CC_file'.format(p.CC_dir)
        p.ref_ang_file = '{}/topos/Euler_PrD/PrD_map.txt'.format(p.out_dir)
        p.ref_ang_file1 = '{}/topos/Euler_PrD/PrD_map1.txt'.format(p.out_dir)


sys.modules[__name__].__class__ = Params
