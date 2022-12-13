from setuptools import find_packages, setup
from glob import glob

classes = """
    Development Status :: 4 - Beta
    License :: OSI Approved :: GNU General Public License v3 (GPLv3)
    Topic :: Software Development :: Libraries
    Topic :: Software Development :: User Interfaces
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Image Processing
    Topic :: Scientific/Engineering :: Visualization
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Determination of conformational continua of macromolecules from single-particle cryo-EM data.')


setup(name='ManifoldEM',
      version='0.2.0',
      description=description,
      author_email="evan.e.seitz@gmail.com",
      maintainer_email="evan.e.seitz@gmail.com",
      packages=find_packages(),
      install_requires=[
          'numpy',
          'mayavi',
          'PyQt5',
          'psutil',
          'matplotlib',
          'scikit-learn',
          'scipy',
          'pandas',
          'mrcfile',
          'opencv-python-headless',
          'opencv-contrib-python',
          'mpi4py',
          'h5py',
          'imageio',
      ],
      # scripts=glob('scripts/*'),
      classifiers=classifiers,
      # package_data={
      #     'deepblast': ['pretrained_models/lstm2x.pt'],
      # }
      )
