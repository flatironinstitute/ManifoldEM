# ManifoldEM

Note this is the Flatiron Institute fork of the [original
ManifoldEM](https://github.com/evanseitz/ManifoldEM_Python).  This particular fork is
maintained by [Robert Blackwell](https://github.com/blackwer) and [Sonya
Hanson](https://github.com/sonyahanson). This work has made significant contributions in
refactoring, code cleanup, optimization, portability, standardization, and distribution since
the fork.

## ManifoldEM Python Suite

This repository contains the Python software implementation of ManifoldEM for determination of
conformational continua of macromolecules from single-particle cryo-EM data, as was first
introduced by Dashti, et al. (2014). A detailed user manual is provided
[here](tutorial/README.md).  Carefully going through this manual will prepare you for running
ManifoldEM on your own data sets. If you have any questions about ManifoldEM after reading this
entire document, carefully check this GitHub forum for similar inquiries or, if no similar
posts exist, create a new thread detailing your inquiry.

This software was initially developed in the Frank research group at Columbia University
(https://joachimfranklab.org) in collaboration with members from UWM (see below). The following
resources may prove useful for a review of ManifoldEM history, theory and implementations:
1. Dashti, A. et al. Trajectories of the ribosome as a Brownian nanomachine. PNAS, 2014.
2. Dashti, A. et al. Retrieving functional pathways of biomolecules from single-particle
   snapshots. Nature Communications, 2020.
3. Mashayekhi, G. ManifoldEM Matlab repository. https://github.com/GMashayekhi/ManifoldEM_Matlab
4. Seitz, E. et al. Geometric machine learning informed by ground truth: Recovery of
   conformational continuum from single-particle cryo-EM data of biomolecules. bioRxiv, 2021.

## Installation
Should be installable in any modern python/conda environment (python 3.7+, though `mayavi` and
`pyqt` packages don't always immediately work with the most recent version of python).

python:
```bash
# create virtual environment. feel free to change the path!
python3 -m venv ~/envs/manifoldem
source ~/envs/manifoldem/activate

pip install --upgrade pip
pip install "git+ssh://git@github.com/flatironinstitute/ManifoldEM"

manifold-gui
```

conda:
```bash
conda create -n manifoldem
conda activate manifoldem

conda install mayavi pyqt=5 python=3.10 -c conda-forge
pip install "git+ssh://git@github.com/flatironinstitute/ManifoldEM"

manifold-gui
```

Note that when using conda, this bypasses conda's package management system and can lead to
problems if you later install packages into this environment with `conda install`. It's
recommended to keep an environment purely for `ManifoldEM`.


## Running without 3D acceleration
Some environments might not allow hardware 3D acceleration, such as via X forwarding or most
VNC/virtual desktop environments. To work around this, you can disable any 3D visualization
widgets in the GUI. This can be done by setting the environment variable `MANIFOLD_DISABLE_VIZ`
to anything 'truthy'. I.e.
```bash
MANIFOLD_DISABLE_VIZ=1 manifold-gui
```


### Contributions
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
If you find this code useful in your work, please cite

{E. Seitz *et al.*, "ManifoldEM Python repository," *Zenodo*, 2021, doi: 10.5281/zenodo.5578874}

[![DOI](https://zenodo.org/badge/405477119.svg)](https://zenodo.org/badge/latestdoi/405477119)


### License
ManifoldEM Copyright (C) 2020-2023 Robert Blackwell, Ali Dashti, Joachim Frank, Sonya Hanson,
Hstau Liao, Suvrajit Maji, Ghoncheh Mashayekhi, Abbas Ourmazd, Peter Schwander, Evan Seitz

The software, code sample and their documentation made available on this repository could
include technical or other mistakes, inaccuracies or typographical errors. We may make changes
to this software or documentation at any time without prior notice, and we assume no
responsibility for errors or omissions therein.

For further details, please see the LICENSE file.
