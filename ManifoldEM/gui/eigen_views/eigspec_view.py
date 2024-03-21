import os
import csv

import numpy as np

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ManifoldEM.params import params


class _EigenSpectrumWindow(QMainWindow):
    def __init__(self):
        super(_EigenSpectrumWindow, self).__init__()

        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)
        self.eigval_canvas = _EigenSpectrumCanvas(self, width=5, height=4)
        toolbar = NavigationToolbar(self.eigval_canvas, self)
        vbl = QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(self.eigval_canvas)
        self.show()


    def plot(self, index: int):
        self.eigval_canvas.plot(index)


class _EigenSpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)


    def load_eigvals(self, index: int):
        # all eigenvecs/vals:
        self.eig_n = []
        self.eig_v = []
        # user-computed vecs/vals (color blue):
        self.eig_n1 = []
        self.eig_v1 = []
        # remaining vecs/vals via [eig_n - eig_n1] (color gray):
        self.eig_n2 = []
        self.eig_v2 = []

        fname = os.path.join(params.out_dir, 'topos', f'PrD_{index}', 'eig_spec.txt')
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
        col1 = data[0]
        col2 = data[1]
        cols = np.column_stack((col1, col2))

        for i, j in cols:
            self.eig_n.append(int(i))
            self.eig_v.append(float(j))
            if int(i) <= int(params.num_psi):
                self.eig_n1.append(int(i))
                self.eig_v1.append(float(j))
            else:
                self.eig_n2.append(int(i))
                self.eig_v2.append(float(j))


    def plot(self, index: int):
        self.load_eigvals(index)

        self.axes.clear()
        self.axes.bar(self.eig_n1, self.eig_v1, edgecolor='none', color='#1f77b4', align='center')  #C0: blue
        self.axes.bar(self.eig_n2, self.eig_v2, edgecolor='none', color='#7f7f7f', align='center')  #C7: gray

        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.axes.set_title('Eigenvalue Spectrum', fontsize=8)
        self.axes.set_xlabel(r'$\mathrm{\Psi}$', fontsize=8)
        self.axes.set_ylabel(r'$\mathrm{\lambda}$', fontsize=8, rotation=0)
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.axhline(0, color='k', linestyle='-', linewidth=.25)
        self.axes.get_xaxis().set_tick_params(direction='out', width=.25, length=2)
        self.axes.get_yaxis().set_tick_params(direction='out', width=.25, length=2)
        self.axes.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        self.axes.yaxis.offsetText.set_fontsize(6)
        self.axes.set_xticks(self.eig_n)
        self.axes.autoscale()
        self.draw()
