import os

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from ManifoldEM.params import params

# FIXME: Merge these two...
class _ClusterAvgMain(QMainWindow):
    def __init__(self, img):
        super(_ClusterAvgMain, self).__init__()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)
        canvas = _ClusterAvgCanvas(self, img, width=2, height=2)
        toolbar = NavigationToolbar(canvas, self)
        vbl = QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(canvas)
        self.show()


class _ClusterAvgCanvas(FigureCanvas):
    def __init__(self, parent, img, width=2, height=2, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.img = img

        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        self.plot()


    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_title('Image Average', fontsize=6)
        ax.imshow(self.img, cmap='gray')
        ax.axis('off')
        self.draw()


class _AverageViewWindow(QMainWindow):
    def __init__(self, index: int):
        super(_AverageViewWindow, self).__init__()
        self.left = 10
        self.top = 10

        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)
        self.image = _AverageViewCanvas(self, width=2, height=2)
        toolbar = NavigationToolbar(self.image, self)
        vbl = QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(self.image)
        self.show()


    def plot(self, index: int):
        self.image.plot(index)


class _AverageViewCanvas(FigureCanvas):
    def __init__(self, parent=None, index=1, width=2, height=2, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        self.plot(index)


    def plot(self, index: int):
        fname = os.path.join(params.out_dir, 'topos', f'PrD_{index}', 'class_avg.png')
        img = mpimg.imread(fname)
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_title('2D Class Average', fontsize=6)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        self.draw()
