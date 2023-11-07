import pickle

import numpy as np

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QMessageBox

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as
                                                NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from ManifoldEM.data_store import data_store

from ManifoldEM.params import p


class _BandwidthMain(QMainWindow):

    def __init__(self):
        super(_BandwidthMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&Kernel Bandwidth', self.guide_bandwidth)

    def guide_bandwidth(self):
        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Help')
        box.setText('<b>Gaussian Kernel Bandwidth</b>')
        box.setInformativeText("<span style='font-weight:normal;'>\
                                A log-log plot of the sum of the elements of the pairwise similarity matrix \
                                as a function of the Gaussian kernel bandwidth.\
                                <br /><br />\
                                The linear region delineates\
                                the range of suitable epsilon values. Twice its slope provides an estimate\
                                of the effective dimensionality.\
                                </span>")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()

    def initUI(self):
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)
        self.bw_canvas = _BandwidthCanvas(self, width=5, height=4)
        toolbar = NavigationToolbar(self.bw_canvas, self)
        vbl = QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(self.bw_canvas)
        self.show()

    def plot(self, index: int):
        self.bw_canvas.plot(index)


class _BandwidthCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)

    def plot(self, index: int):
        ds = data_store.get_diff_maps()
        logEps = ds.log_eps(index)
        logSumWij = ds.log_sum_Wij(index)
        popt = data.popt(index)
        R_squared = data.R_squared(index)

        def fun(xx, aa0, aa1, aa2, aa3):  #fit tanh()
            #aa3: y-value of tanh inflection point
            #aa2: y-value of apex (asymptote)
            #aa1: x-value of x-shift (inverse sign)
            #aa0: alters spread
            F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
            return F

        self.axes.clear()
        self.axes.scatter(logEps, logSumWij, s=1, c='C0', edgecolor='C0', zorder=.1, label='data')
        self.axes.plot(logEps,
                       fun(logEps, popt[0], popt[1], popt[2], popt[3]),
                       c='C1',
                       linewidth=.5,
                       zorder=.2,
                       label=r'$\mathrm{tanh(x)}$')
        self.axes.axvline(-(popt[1] / popt[0]),
                          color='C2',
                          linewidth=.5,
                          linestyle='-',
                          zorder=0,
                          label=r'$\mathrm{ln \ \epsilon}$')
        self.axes.plot(logEps,
                       popt[0] * popt[2] * (logEps + popt[1] / popt[0]) + popt[3],
                       c='C3',
                       linewidth=.5,
                       zorder=.3,
                       label='slope')
        self.axes.set_ylim(
            np.amin(fun(logEps, popt[0], popt[1], popt[2], popt[3])) - 1,
            np.amax(fun(logEps, popt[0], popt[1], popt[2], popt[3])) + 1)
        self.axes.legend(loc='lower right', fontsize=6)

        slope = popt[0] * popt[2]  #slope of tanh

        textstr = '\n'.join((
            r'$y=%.2f + %.2f tanh(%.2fx + %.2f)$' % (popt[3], popt[2], popt[0], popt[1]),
            r'$\mathrm{Slope=%.2f}$' % (slope, ),
            r'$\mathrm{Optimal \ log(\epsilon)=%.2f}$' % (-(popt[1] / popt[0]), ),
            r'$\mathrm{R^2}=%.2f$' % (R_squared, ),
        ))

        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

        self.axes.text(0.05,
                       0.95,
                       textstr,
                       transform=self.axes.transAxes,
                       fontsize=6,
                       verticalalignment='top',
                       bbox=props)

        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)

        self.axes.set_title('Gaussian Kernel Bandwidth', fontsize=8)
        self.axes.set_xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=8)
        self.axes.set_ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=8, rotation=90)
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.yaxis.offsetText.set_fontsize(6)
        self.axes.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.draw()
