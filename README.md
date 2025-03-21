# ManifoldEM

Note this is the Flatiron Institute fork of the [original Python
ManifoldEM](https://github.com/evanseitz/ManifoldEM_Python).  This particular fork is
maintained by [Robert Blackwell](https://github.com/blackwer) and [Sonya
Hanson](https://github.com/sonyahanson). This work has made significant contributions in
refactoring, code cleanup, optimization, portability, standardization, and distribution since
the fork.

## ManifoldEM Python Suite

This repository contains the Python software implementation of ManifoldEM for determination of
conformational continua of macromolecules from single-particle cryo-EM data, as was first
introduced by [Dashti, et al. (2014)](https://doi.org/10.1073/pnas.1419276111). A detailed user manual is provided
[here](tutorial/README.md).  Carefully going through this manual will prepare you for running
ManifoldEM on your own data sets. If you have any questions about ManifoldEM after reading this
entire document, carefully check this GitHub forum for similar inquiries or, if no similar
posts exist, create a new thread detailing your inquiry.

This software was initially developed in the [Frank lab](https://joachimfranklab.org) at Columbia University in collaboration with members from UWM (see below). The following resources may prove useful for a review of ManifoldEM history, theory and implementations:
1. Dashti, A. et al. Trajectories of the ribosome as a Brownian nanomachine. PNAS, 2014. [DOI](https://doi.org/10.1073/pnas.1419276111)
2. Dashti, A. et al. Retrieving functional pathways of biomolecules from single-particle snapshots. Nature Communications, 2020. [DOI](https://doi.org/10.1038/s41467-020-18403-x)
3. Mashayekhi, G. ManifoldEM Matlab repository. https://github.com/GMashayekhi/ManifoldEM_Matlab
4. Maji, S. et al. Propagation of Conformational Coordinates Across Angular Space in Mapping the Continuum of States from Cryo-EM Data by Manifold Embedding. JCIM, 2020. [DOI](https://doi.org/10.1021/acs.jcim.9b01115)
5. Seitz, E. et al. Recovery of conformational continuum from single-particle cryo-EM images: Optimization of manifoldEM informed by Ground Truth. IEEE Transactions on Computational Imaging, 2022. [DOI](https://doi.org/10.1109/TCI.2022.3174801)


## Installation
Should be installable in any modern python/conda environment (python 3.9+, though `mayavi` and
`pyqt` packages don't always immediately work with the most recent version of python). If you don't
need the gui, feel free to omit the "[gui]" part of the install command!

python:
```bash
# create virtual environment. feel free to change the path!
python3 -m venv ~/envs/manifoldem
source ~/envs/manifoldem/bin/activate

pip install --upgrade pip
pip install "manifoldem[gui] @ git+https://github.com/flatironinstitute/ManifoldEM"

manifold-gui
```

conda:
```bash
conda create -n manifoldem python=3.10 -y
conda activate manifoldem

pip install "manifoldem[gui] @ git+https://github.com/flatironinstitute/ManifoldEM"

manifold-gui
```

Note that when using conda, this bypasses conda's package management system and can lead to
problems if you later install packages into this environment with `conda install`. It's
recommended to keep an environment purely for `ManifoldEM`.


## Running without 3D acceleration
Some environments might not allow hardware 3D acceleration, such as via X forwarding or most
VNC/virtual desktop environments. To work around this, you can disable any 3D visualization
widgets in the GUI. This can be done by setting the environment variable by providing the `-V`
flag to `manifold-gui`

```bash
manifold-gui -V
```


## Basic command line interface
For most steps in the ManifoldEM pipeline, the GUI is unnecessary and sometimes even burdensome. For
this reason we supply a basic command line interface. The CLI allows the user to make more granular
steps through the analysis pipeline and is generally more useful for cluster/remote environment and
debugging. All CLI invocations start with the program `manifold-cli`. If you run `manifold-cli` with
no arguments, it will print a help message and exit.

```
% manifold-cli
ManifoldEM version: 0.3.1.dev60+ga73affd.d20241113

usage: manifold-cli [-h] [-n NCPU] {init,threshold,calc-distance,manifold-analysis,psi-analysis,nlsa-movie,find-ccs,calc-probabilities,trajectory,utility} ...

Command-line interface for ManifoldEM package

positional arguments:
  {init,threshold,calc-distance,manifold-analysis,psi-analysis,nlsa-movie,find-ccs,calc-probabilities,trajectory,utility}
    init                0: Initialize new project
    threshold           1: Set upper/lower thresholds for principal direction detection
    calc-distance       2: Calculate S2 distances
    manifold-analysis   3: Initial embedding
    psi-analysis        4: Analyze images to get psis
    nlsa-movie          5: Create 2D psi movies
    find-ccs            7: Find conformational coordinates
    calc-probabilities  8: Calculate probability landscape
    trajectory          9: Calculate trajectory
    utility             Utility functions

options:
  -h, --help            show this help message and exit
  -n NCPU, --ncpu NCPU
```

The output shows that there are nine sub-commands listed in the order they belong in the
pipeline (and some utility functions). Some commands support additional arguments, especially
the `init` command, which creates a new project in your current working directory. To see how
to use a given command, simply run the command with a following `-h` flag, e.g.

```
% manifold-cli init -h
ManifoldEM version: 0.3.1.dev60+ga73affd.d20241113

usage: manifold-cli init [-h] -p STR [-v FILEPATH] [-a FILEPATH] [-i FILEPATH] [-m FILEPATH] -s FLOAT -d FLOAT -r FLOAT [-x INT] [-o] [--eps FLOAT] [--prd_thres_low INT]
                         [--prd_thres_high INT] [--tess_hemisphere_vec STR] [--tess_hemisphere_type STR] [--distance_filter_type STR] [--distance_filter_cutoff_freq FLOAT]
                         [--distance_filter_order INT] [--num_psi INT] [--nlsa_tune INT] [--con_order_range INT] [--nlsa_fps FLOAT]

options:
  -h, --help            show this help message and exit
  -p STR, --project-name STR
                        Name of project to create (default: None)
  -v FILEPATH, --avg-volume FILEPATH
  -a FILEPATH, --alignment FILEPATH
  -i FILEPATH, --image-stack FILEPATH
  -m FILEPATH, --mask-volume FILEPATH
  -s FLOAT, --pixel-size FLOAT
  -d FLOAT, --diameter FLOAT
  -r FLOAT, --resolution FLOAT
  -x INT, --aperture-index INT
  -o, --overwrite       Replace existing project with same name automatically (default: False)
  --eps FLOAT           [BINNING] Small fraction to be added if divide-by-zero errors occur (default: 1e-10)
  --prd_thres_low INT   [BINNING] Minimum required snapshots in a tessellation for it be admitted (default: 100)
  --prd_thres_high INT  [BINNING] Maximum number of snapshots that will be considered within each tessellation (default: 2000)
  --tess_hemisphere_vec STR
                        [BINNING] Vector perpendicular to the plane defining which half of S2 (image viewing directions) to place PrDs. PrDs opposite this plane will be
                        mirrored (default: [1.0, 0.0, 0.0])
  --tess_hemisphere_type STR
                        [BINNING] Technique to tesselate sphere. Valid options: ["lovisolo_silva", "fibonacci"] (default: lovisolo_silva)
  --distance_filter_type STR
                        [CALC_DISTANCE] Filter type for image preprocessing. Valid: {"Butter", "Gauss"} (default: Butter)
  --distance_filter_cutoff_freq FLOAT
                        [CALC_DISTANCE] Nyquist cutoff frequency for filter (default: 0.5)
  --distance_filter_order INT
                        [CALC_DISTANCE] Order of Filter ("Butter" only) (default: 8)
  --num_psi INT         [CALC_DISTANCE] Number of eigenfunctions for analysis (default: 8)
  --nlsa_tune INT       [MANIFOLD_ANALYSIS] Diffusion map tuning parameter (default: 3)
  --con_order_range INT
                        [PSI_ANALYSIS] Coarse-graining factor of probability landscape (default: 50)
  --nlsa_fps FLOAT      [NLSA_MOVIE] Frames per second for generated movies (default: 5.0)
```

An example invocation then might look like

```
manifold-cli init -v J310/J310_003_volume_map.mrc -a J310/from_csparc.star -i J310/signal_subtracted.mrcs -s 1.22 -d 160.0 -r 3.02 -x 1 -p my_J310_analysis
```

The rest of the commands will take as their final argument the "toml" file generated from the
initialization step. Let's set the image count thresholds and calculate the distance matrices for
the leading five eigenvalue decompositions ("psis") next as an example. Note the `-n 16` _before_
the sub-command. Most processing steps support this option, which specifies how many workers to use
in processing. In most cases you want roughly the output of the `nproc` command. My workstation has
16 physical cores, so I specify 16 below for the matrix calculation step. Supplying `-n` for
commands that don't support parallel processing is harmless.

```
% manifold-cli threshold --prd_thres_low 100 --prd_thres_high 4000 params_my_J310_analysis.toml
ManifoldEM version: 0.3.1.dev60+ga73affd.d20241113

Changing param prd_thres_high from 2000 to 4000
% manifold-cli -n 16 calc-distance --num_psi 5 params_my_J310_analysis.toml
ManifoldEM version: 0.3.1.dev60+ga73affd.d20241113

Changing param num_psi from 3 to 5
Computing the distances...
Calculating projection direction information
RELION Optics Group found.
Number of PDs: 145
Neighborhood epsilon: 0.05338763021220811
Number of Graph Edges: (982, 2)

Performing connected component analysis.
Number of connected components: 2
Number of Graph Edges: (518, 2)
Number of Graph Edges: (464, 2)
100%|███████████████████████████████| 145/145 [00:11<00:00, 12.17it/s]
```

This has created a significant amount of data stored in the `output/my_J310_analysis/distances` --
one file for each principal direction. Currently there isn't much tooling to visualize these
outputs, though that is a work in progress. Each file is a python `pickle` file and can be inspected
using the usual python tooling for the curious user.

Let's finish up the first major stage of the pipeline. I'm hiding the output for clarity's sake.

```
% manifold-cli -n 16 manifold-analysis params_my_J310_analysis.toml &> /dev/null
% manifold-cli -n 16 psi-analysis params_my_J310_analysis.toml &> /dev/null
% manifold-cli -n 16 nlsa-movie params_my_J310_analysis.toml &> /dev/null
```

At this point, if you wanted to visualize and manually manipulate the principle directions and
associated data, you could simply `manifold-gui -R params_my_J310_analysis.toml`. Here you could set
the anchor directions, manually control the sense of each direction, remove directions, and other
various things. Once you hit the "Compile Results" command, you can continue using the command
line. Here I set a few anchors and will continue on...

```
% manifold-cli -n 16 find-ccs params_my_J310_analysis.toml &> /dev/null
% manifold-cli -n 16 calc-probabilities params_my_J310_analysis.toml &> /dev/null
% manifold-cli -n 16 trajectory params_my_J310_analysis.toml &> /dev/null
```


### Python/Jupyter interface
Researchers/the curious are possibly interested in various things "under the hood" in
`ManifoldEM`. We've provided a basic interface for accessing some of the internal data
calculated/written during various stages of the `ManifoldEM` pipeline. Most internal state
isn't fully documented, and is likely to change, so it might take some sleuthing/educated
guesswork to figure out what you're actually looking at, and that might disappear in future
versions. Regardless, all exposed data can be accessed via the `data_store` `API`, which is
documented and accessible via the `python` `help` interface. An example ipython session...

```
In [1]: from ManifoldEM.data_store import data_store
   ...: from ManifoldEM.params import params
   ...: params.load('params_J310.toml')
   ...: prd = data_store.get_prd_data(117)
   ...: npix = params.ms_num_pixels
   ...: print(prd)
prd_index: 117
S2_bin_index: 17310
bin_center: [ 0.01355376 -0.00546575  0.9998932 ]
n_images: 216
occupancy: 216
trash: False
anchor: True
cluster_id: 0

In [2]: help(prd)
...
class PrdData(builtins.object)
 |  PrdData(prd_index: int)
 |
 |  Represents a single projection direction, providing access to its raw and transformed images, CTF images, and metadata.
 |
 |  Attributes
 |  ----------
 |  info : PrdInfo
 |      Metadata about the projection direction.
 |  raw_images : ndarray
 |      The raw images from the image stack associated with the projection direction.
 |  transformed_images : ndarray
 |      The filtered and "in-plane" rotated images associated with the projection direction.
 |  ctf_images : ndarray
 |      The Contrast Transfer Function (CTF) images associated with the projection direction.
 |  psi_data : dict
 |      The embedding data associated with the projection direction.
 |  EL_data : dict
 |      The NLSA/eigenvalue data associated with the projection direction.
 |  dist_data : dict
 |      The distance information between images in the projection direction, including transformed images
 |      in the `transformed_images` attribute.
 |
 |  ...
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |
 |  EL_data
 |
 |  ctf_images
 |
 |  info
 |
 |  psi_data
 |
 |  raw_images
 |
 |  transformed_images
 |
 |  ----------------------------------------------------------------------
...

In [3]: import matplotlib.pyplot as plt

In [4]: plt.imshow(prd.EL_data['IMG1'][:,-1].reshape(npix, npix))
Out[4]: <matplotlib.image.AxesImage at 0x7fde924838b0>

In [5]: plt.show()

In [6]: prd.EL_data.keys()
Out[6]: dict_keys(['IMG1', 'IMGT', 'posPath', 'PosPsi1', 'psirec', 'tau', 'psiC1', 'mu', 'VX', 'sdiag', 'Topo_mean', 'tauinds'])

In [7]: prd.psi_data.keys()
Out[7]: dict_keys(['lamb', 'psi', 'sigma', 'mu', 'posPath', 'ind', 'logEps', 'logSumWij', 'popt', 'R_squared'])

In [8]: prd.raw_images.shape
Out[8]: (216, 192, 192)

In [9]: plt.imshow(prd.raw_images[10]); plt.show()

In [10]: plt.imshow(prd.transformed_images[10]); plt.show()

In [11]: prd.transformed_images.shape
Out[11]: (216, 192, 192)
```


### Contributions

Current ManifoldEM Python team (alphabetically ordered):

- Miro A. Astore, Flatiron Institute
- Robert Blackwell, Flatiron Institute
- Eduard Cruz-Chu, University of Wisconsin-Milwaukee
- Raison Dsouza, University of Wisconsin-Madison
- Sonya M. Hanson, Flatiron Institute
- Anand A. Ojha, Flatiron Institute
- Peter Schwander, University of Wisconsin-Milwaukee

Original ManifoldEM Python team (alphabetically ordered):

- Ali Dashti, University of Wisconsin-Milwaukee
- Joachim Frank, Columbia University
- Hstau Liao, Columbia University
- Suvrajit Maji, Columbia University
- Ghoncheh Mashayekhi, University of Wisconsin-Milwaukee
- Abbas Ourmazd, University of Wisconsin-Milwaukee
- Peter Schwander, University of Wisconsin-Milwaukee
- Evan Seitz, Columbia University

The original individual author contributions are usually provided in the headers of each source
file, or in the functions. While reasonable effort has been made to retain copyright notices
for individual contributions from the source material, significant refactorings have made some
individual contributions hard to track or ultimately meaningless.


### Attribution
If you find this code useful in your work, please cite:

A. Ojha et al., The ManifoldEM method for cryo-EM: a step-by-step breakdown accompanied by a modern Python implementation. *Acta Cryst.* (2025) D81, 89-104. [DOI](https://doi.org/10.1107/S2059798325001469)




### License
ManifoldEM Copyright (C) 2020-2025 Robert Blackwell, Ali Dashti, Joachim Frank, Sonya Hanson,
Hstau Liao, Suvrajit Maji, Ghoncheh Mashayekhi, Abbas Ourmazd, Peter Schwander, Evan Seitz

The software, code sample and their documentation made available on this repository could
include technical or other mistakes, inaccuracies or typographical errors. We may make changes
to this software or documentation at any time without prior notice, and we assume no
responsibility for errors or omissions therein.

For further details, please see the LICENSE file.
