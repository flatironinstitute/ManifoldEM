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
from ManifoldEM.params import p
from ManifoldEM.gui import MainWindow

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


def init(args):
    if (args.restore):
        p.load(args.restore)
        return

    p.proj_name = args.project_name

    p.avg_vol_file = args.avg_volume
    p.align_param_file = args.alignment
    p.img_stack_file = args.image_stack
    p.mask_vol_file = args.mask_volume

    p.pix_size = args.pixel_size
    p.obj_diam = args.diameter
    p.resol_est = args.resolution
    p.ap_index = args.aperture_index


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
    if init(args):
        sys.exit(1)

    app = QApplication([])
    app.setStyle('fusion')
    app.setFont(QFont('Arial', 12))

    QtCore.QCoreApplication.setApplicationName('ManifoldEM')

    w = MainWindow()
    w.setWindowTitle('ManifoldEM')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()