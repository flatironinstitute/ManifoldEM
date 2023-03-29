import os
import sys

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

proj_name: str = ''            # name of the project :D
resProj: int = 0               # see above
relion_data: bool = False      # working with relion data?
ncpu: int = 1                  # number of CPUs for multiprocessing
machinefile: str = ''          # machinefile for MPI
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
num_psiTrunc:int = 8    # number of eigenfunctions for truncated views
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
temperature: float = 25.0 # user-defined pre-quenching temperature of experiments
trajName: str = '1'       # filename for exported (2D) trajectories
nClass: int = 50          # number of states partitioned within each 1D reaction coordinate; results in a 50x1 1D ELS
width_1D: int = 1         # user-defined width of trajectory in 1D energy path
width_2D: int = 1         # user-defined width of trajectory in 2D energy path
hUh = None                # occupancy map


#  reaction coordinates parameters:
getOpticalFlow: bool = True                # default True to compute optical flow vectors
getAllEdgeMeasures: bool = True            # default True to compute edge measures
anch_list: list[int] = []                  # user-defined PD anchors for Belief Propagation
trash_list: list[int] = []                 # user-defined PD removals to ignore via final compile [binary list, 1 entry/PD]
opt_movie: dict = {'printFig': 0,
                   'OFvisual': 0,
                   'visual_CC': 0,
                   'flowVecPctThresh': 95}
opt_mask_type: int = 0                     # 0:None, 1:Annular, 2:Volumetric
opt_mask_param: int = 0                    # for either none, radius (Int), or iso(Int)

def todict():
    res = {}
    module = sys.modules[__name__]
    for var in dir(module):
        if not var.startswith('_') and isinstance(getattr(module, var), (int, list, dict, float)):
            res[var] = getattr(module, var)

    return res

def create_dir():
    # input and output directories and files
    global user_dir,dist_dir,dist_prog,psi_dir,psi_prog,psi2_dir,psi2_prog,\
        movie2d_dir,EL_dir,EL_prog,\
        tau_dir,OM_dir,Var_dir,NLSA_dir,traj_dir,bin_dir,relion_dir,\
        CC_dir,CC_OF_dir, CC_meas_dir, CC_meas_prog, out_dir, \
        post_dir, vol_dir, svd_dir, anim_dir

    dist_dir = os.path.join(user_dir, 'outputs_{}/distances/'.format(proj_name))
    dist_prog = os.path.join(dist_dir, 'progress/')
    os.makedirs(dist_prog, exist_ok=True)

    psi_dir = os.path.join(user_dir, 'outputs_{}/diff_maps/'.format(proj_name))
    psi_prog = os.path.join(psi_dir, 'progress/')
    os.makedirs(psi_prog, exist_ok=True)

    psi2_dir = os.path.join(user_dir, 'outputs_{}/psi_analysis/'.format(proj_name))
    psi2_prog = os.path.join(psi2_dir, 'progress/')
    os.makedirs(psi2_prog, exist_ok=True)

    EL_dir = os.path.join(user_dir, 'outputs_{}/ELConc{}/'.format(proj_name, conOrderRange))
    EL_prog = os.path.join(EL_dir, 'progress/')
    os.makedirs(EL_prog, exist_ok=True)

    OM_dir = os.path.join(user_dir, '{}OM/'.format(EL_dir))
    os.makedirs(OM_dir, exist_ok=True)

    Var_dir = os.path.join(user_dir, 'outputs_{}/Var/'.format(proj_name))
    os.makedirs(Var_dir, exist_ok=True)
    traj_dir = os.path.join(user_dir, 'outputs_{}/traj/'.format(proj_name))
    os.makedirs(traj_dir, exist_ok=True)

    relion_dir = bin_dir = os.path.join(user_dir, 'outputs_{}/bin/'.format(proj_name))
    os.makedirs(bin_dir, exist_ok=True)

    CC_dir = os.path.join(user_dir, 'outputs_{}/CC/'.format(proj_name))
    CC_OF_dir = os.path.join(CC_dir, 'CC_OF')
    os.makedirs(CC_OF_dir, exist_ok=True)

    CC_meas_dir = os.path.join(CC_dir, 'CC_meas')
    CC_meas_prog = os.path.join(CC_meas_dir, 'progress')
    os.makedirs(CC_meas_prog, exist_ok=True)

    #################
    # post-processing:
    post_dir = os.path.join(user_dir, 'outputs_{}/post/'.format(proj_name))
    vol_dir = os.path.join(post_dir, '1_vol')
    svd_dir = os.path.join(post_dir, '2_svd')
    anim_dir = os.path.join(post_dir, '3_anim')
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(vol_dir, exist_ok=True)
    os.makedirs(svd_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    #################
    out_dir = os.path.join(user_dir, 'outputs_{}/'.format(proj_name))
    os.makedirs(os.path.join(out_dir, 'topos', 'Euler_PrD'), exist_ok=True)

    global dist_file,psi_file,psi2_file,\
        movie2d_file,EL_file,tau_file,OM_file,OM1_file,Var_file,rho_file,\
        remote_file,NLSA_file,traj_file,CC_file,CC_OF_file,CC_meas_file,\
        CC_graph_file,ref_ang_file,ref_ang_file1,tess_file,nowTime_file

    tess_file = os.path.join(user_dir, 'outputs_{}/selecGCs'.format(proj_name))
    nowTime_file = os.path.join(user_dir, 'outputs_{}/nowTime'.format(proj_name))
    dist_file = '{}/IMGs_'.format(dist_dir)
    psi_file = '{}/gC_trimmed_psi_'.format(psi_dir)
    psi2_file = '{}/S2_'.format(psi2_dir)
    EL_file = '{}/S2_'.format(EL_dir)
    OM_file = '{}/S2_'.format(OM_dir)
    OM1_file = '{}/S2_'.format(OM_dir)
    Var_file = '{}/S2_'.format(Var_dir)
    rho_file = '{}/rho'.format(OM_dir)
    remote_file = '{}/rem_'.format(Var_dir)
    traj_file = '{}/traj_'.format(traj_dir)
    CC_graph_file = '{}graphCC'.format(CC_dir)
    CC_OF_file = '{}OF_prD_'.format(CC_OF_dir)
    CC_meas_file = '{}meas_edge_prDs_'.format(CC_meas_dir)
    CC_file = '{}CC_file'.format(CC_dir)
    ref_ang_file = '{}/topos/Euler_PrD/PrD_map.txt'.format(out_dir)
    ref_ang_file1 = '{}/topos/Euler_PrD/PrD_map1.txt'.format(out_dir)
