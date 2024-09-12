import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'
# Workaround for mayavi + qt5 (tvtk actually?)
os.environ['ETS_TOOLKIT'] = 'qt4'

import sys
import time

from argparse import ArgumentParser

from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

import ManifoldEM

parser = ArgumentParser(
    prog="manifold-gui",
    description="GUI interface for ManifoldEM package",
)

parser.add_argument('-p', "--project-name", type=str,
                    help="Name of project to create",
                    default=time.strftime("%Y%m%d_%H%M%S"))
parser.add_argument('-v', "--avg-volume", type=str, default="")
parser.add_argument('-a', "--alignment", type=str, default="")
parser.add_argument('-i', "--image-stack", type=str, default="")
parser.add_argument('-m', "--mask-volume", type=str, default="")
parser.add_argument('-s', "--pixel-size", type=float, default=1.0)
parser.add_argument('-d', "--diameter", type=float, default=150.0)
parser.add_argument('-r', "--resolution", type=float, default=1.0)
parser.add_argument('-x', "--aperture-index", type=int, default=1)
parser.add_argument('-R', "--restore", type=str, default="")
parser.add_argument('-V', "--disable-viz", action='store_true')


def init(args):
    from ManifoldEM.params import params
    if args.disable_viz:
        os.environ['MANIFOLD_DISABLE_VIZ'] = '1'

    if (args.restore):
        params.load(args.restore)
        return

    params.project_name = args.project_name

    params.avg_vol_file = args.avg_volume
    params.align_param_file = args.alignment
    params.img_stack_file = args.image_stack
    params.mask_vol_file = args.mask_volume

    params.ms_pixel_size = args.pixel_size
    params.particle_diameter = args.diameter
    params.ms_estimated_resolution = args.resolution
    params.aperture_index = args.aperture_index


def main():
    print(r"""
 __  __             _  __       _     _ _____ __  __
|  \/  | __ _ _ __ (_)/ _| ___ | | __| | ____|  \/  |
| |\/| |/ _` | '_ \| | |_ / _ \| |/ _` |  _| | |\/| |
| |  | | (_| | | | | |  _| (_) | | (_| | |___| |  | |
|_|  |_|\__,_|_| |_|_|_|  \___/|_|\__,_|_____|_|  |_|

""")
    print(f"version: {ManifoldEM.__version__}\n")
    args = parser.parse_args(sys.argv[1:])
    init(args)

    app = QApplication([])
    app.setStyle('fusion')
    app.setFont(QFont('Arial', 12))

    QtCore.QCoreApplication.setApplicationName('ManifoldEM')

    from ManifoldEM.gui import MainWindow
    w = MainWindow()
    w.setWindowTitle('ManifoldEM')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
