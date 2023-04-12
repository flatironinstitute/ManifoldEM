# README

Note this is the Flatiron Institute hosted version of this code.  Further development is
currently being conducted by Robert Blackwell and Sonya Hanson -- feel free to reach out
directly with any questions or concerns, or file an issue.

## ManifoldEM Python Suite

This repository contains the Python software implementation of ManifoldEM for determination of conformational continua of macromolecules from single-particle cryo-EM data, as was first introduced by Dashti, et al. (2014). A detailed user manual is provided on the main branch of this repository. Carefully going through this manual will prepare you for running ManifoldEM on your own data sets. If you have any questions about ManifoldEM after reading this entire document, carefully check the ManifoldEM GitHub forum for similar inquiries or, if no similar posts exist, create a new thread detailing your inquiry. 

This software was initially developed in the Frank research group at Columbia University (https://joachimfranklab.org) in collaboration with members from UWM (see below). The following resources may prove useful for a review of ManifoldEM history, theory and implementations:
1. Dashti, A. et al. Trajectories of the ribosome as a Brownian nanomachine. PNAS, 2014.
2. Dashti, A. et al. Retrieving functional pathways of biomolecules from single-particle snapshots. Nature Communications, 2020.
3. Mashayekhi, G. ManifoldEM Matlab repository. https://github.com/GMashayekhi/ManifoldEM_Matlab
4. Seitz, E. et al. Geometric machine learning informed by ground truth: Recovery of conformational continuum from single-particle cryo-EM data of biomolecules. bioRxiv, 2021.

## Installation
Should be installable in any modern python/conda environment (python 3.7+). Needs a working
`mpi` install as well (a default conda one should be fine). On cluster environments this might
require something like

```bash
module load python openmpi python-mpi
```

Then to install...

```bash
# create virtual environment
# --system-site-packages flag optional, but can reduce size of venv -- useful for some cluster environments
python3 -m venv ~/path/to/venv --system-site-packages
source ~/path/to/venv/bin/activate

pip install git+ssh://git@github.com/flatironinstitute/ManifoldEM

ManifoldEM_GUI
```

### Contributions
ManifoldEM Python team (alphabetically ordered):

- Ali Dashti, University of Wisconsin-Milwaukee
- Joachim Frank, Columbia University
- Hstau Liao, Columbia University
- Suvrajit Maji, Columbia University
- Ghoncheh Mashayekhi, University of Wisconsin-Milwaukee
- Abbas Ourmazd, University of Wisconsin-Milwaukee
- Peter Schwander, University of Wisconsin-Milwaukee
- Evan Seitz, Columbia University

Individual author contributions are usually provided in the headers of each source file, or in the functions.

This particular fork is maintained by Robert Blackwell (@blackwer) and Sonya Hanson
(@sonyahanson). This work has made significant contributions in refactoring, code cleanup,
optimization, portability, standardization, and distribution since the fork. While reasonable
effort has been made to retain copyright notices for individual contributions from the source
material, significant refactorings have made some individual contributions hard to track or
ultimately meaningless.


### Attribution
If you find this code useful in your work, please cite

{E. Seitz *et al.*, "ManifoldEM Python repository," *Zenodo*, 2021, doi: 10.5281/zenodo.5578874}

[![DOI](https://zenodo.org/badge/405477119.svg)](https://zenodo.org/badge/latestdoi/405477119)


### License
ManifoldEM Copyright (C) 2020-2023 Robert Blackwell, Ali Dashti, Joachim Frank, Sonya Hanson,
Hstau Liao, Suvrajit Maji, Ghoncheh Mashayekhi, Abbas Ourmazd, Peter Schwander, Evan Seitz,
Robert Blackwell, Sonya Hanson

The software, code sample and their documentation made available on this repository could
include technical or other mistakes, inaccuracies or typographical errors. We may make changes
to this software or documentation at any time without prior notice, and we assume no
responsibility for errors or omissions therein.

For further details, please see the LICENSE file. 
