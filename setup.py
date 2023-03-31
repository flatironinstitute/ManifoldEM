from setuptools import setup
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
      maintainer_email="rblackwell@flatironinstitute.org",
      packages=['ManifoldEM', 'ManifoldEM/CC'],
      install_requires=[
          'numpy<=1.22',
          'mayavi',
          'PyQt5',
          'psutil',
          'matplotlib',
          'scikit-image',
          'scikit-learn',
          'scipy',
          'pillow',
          'pandas',
          'mrcfile',
          'opencv-python-headless',
          'opencv-contrib-python',
          'mpi4py',
          'h5py',
          'imageio',
          'toml',
      ],
      scripts=glob('scripts/*'),
      classifiers=classifiers,
      data_files=[('share/ManifoldEM/icons', ['icons/200x200.png', 'icons/256x256.png', 'icons/70x70.png'])],
      )
