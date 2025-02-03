import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QTabWidget, QDialog, QLabel,
                             QDoubleSpinBox, QAbstractSpinBox, QPushButton, QGridLayout, QLayout)

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
import matplotlib.cm as colormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FixedLocator

from ManifoldEM.params import params
from ManifoldEM.data_store import data_store

class ThresholdView(QMainWindow):
    def __init__(self):
        super(ThresholdView, self).__init__()
        self.thresh_low = params.prd_thres_low
        self.thresh_high = params.prd_thres_high

        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&PD Thresholding', self.guide_threshold)

    def guide_threshold(self):
        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Help')
        box.setText('<b>PD Thresholding</b>')
        box.setInformativeText("<span style='font-weight:normal;'>\
                                The bar chart shows the number of particles within each\
                                projection direction (PD).\
                                <br /><br />\
                                The angular size of each PD is determined by the parameters set\
                                on the first tab. These parameters can be changed before\
                                proceeding to the next section <i>Embedding</i> tab) to alter the distribution\
                                of particles shown in the S2 plot.\
                                <br /><br />\
                                If a PD has fewer particles\
                                than the number specified by the low threshold parameter,\
                                that PD will be ignored across all future computations.\
                                If a PD has more particles\
                                than the number (<i>n</i>) specified by the high threshold parameter,\
                                only the first <i>n</i> particles under this threshold will be used for that PD.\
                                </span>")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()


    def initUI(self):
        self.thresh_all_tab = ThreshAllCanvas(self)
        self.thresh_polar_tab = ThreshFinalCanvas(self)
        self.unique_occupancy_tab = OccHistCanvas(self)
        thresh_tabs = QTabWidget(self)
        thresh_tabs.addTab(self.thresh_all_tab, 'Edit Thresholds')
        thresh_tabs.addTab(self.thresh_polar_tab, 'Thresholded PDs')
        thresh_tabs.addTab(self.unique_occupancy_tab, 'Occupancy Distribution')

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(thresh_tabs)
        self.setWindowModality(QtCore.Qt.ApplicationModal)  #freezes out parent window
        self.show()

        thresh_tabs.currentChanged.connect(self.onTabChange)  #signal for tab changed via direct click

    def onTabChange(self, i):
        prds = data_store.get_prds()
        prd_mask = prds.thres_ids
        occupancies = prds.occupancy

        if i == 1:  #signals when view switched to tab 2
            bin_centers = prds.bin_centers[:, prd_mask]
            phis = np.arctan2(bin_centers[1, :], bin_centers[0, :]) - np.pi
            thetas = np.arccos(bin_centers[2, :]) * 180. / np.pi

            self.thresh_polar_tab.plot(phis, thetas, prds.occupancy)

        if i == 2:
            n_unique = len(set(occupancies))
            self.unique_occupancy_tab.entry_bins.valueChanged.disconnect()
            self.unique_occupancy_tab.entry_bins.valueChanged.connect(lambda: self.unique_occupancy_tab.change_bins(occupancies))
            self.unique_occupancy_tab.entry_bins.setValue(n_unique // 2)
            self.unique_occupancy_tab.entry_bins.setMaximum(n_unique)
            self.unique_occupancy_tab.entry_bins.setSuffix(f' / {n_unique}')


    def closeEvent(self, ce):  #safety message if user clicks to exit via window button
        if not self.thresh_all_tab.confirmed:
            msg = 'Changes to the thresholding parameters have not been confirmed\
                    on the Edit Thresholds tab (via <i>Update Thresholds</i>).\
                    <br /><br />\
                    Do you want to proceed without saving?'

            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setText('<b>Exit Warning</b>')
            box.setIcon(QMessageBox.Warning)
            box.setInformativeText(msg)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = box.exec_()
            if reply == QMessageBox.Yes:
                self.thresh_all_tab.confirmed = 1
                self.thresh_low = params.prd_thres_low
                self.thresh_high = params.prd_thres_high
                self.close()
            else:
                ce.ignore()


class ThreshAllCanvas(QDialog):
    def __init__(self, parent):
        super(ThreshAllCanvas, self).__init__(parent)
        self.thresh_container = parent

        self.confirmed = False

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(1, 1, 1)

        prds = data_store.get_prds()
        self.occupancy_full = prds.occupancy_full
        n_pds = self.occupancy_full.size
        self.axes.bar(range(n_pds), self.occupancy_full, align='center',
                      edgecolor='none', color='#1f77b4', snap=False)

        self.xlimLo = self.axes.get_xlim()[0]
        self.xlimHi = self.axes.get_xlim()[1]

        self.lineL, = self.axes.plot([], [],
                                     color='#d62728',
                                     linestyle='-',
                                     linewidth=.5,
                                     label='Low Threshold')  #red
        self.lineH, = self.axes.plot([], [],
                                     color='#2ca02c',
                                     linestyle='-',
                                     linewidth=.5,
                                     label='High Threshold')  #green
        x = np.arange(self.xlimLo, self.xlimHi + 1)
        self.lineL.set_data(x, [params.prd_thres_low])
        self.lineH.set_data(x, [params.prd_thres_high])

        self.axes.axvline(n_pds + 1, color='#7f7f7f', linestyle='-', linewidth=.5)

        #self.axes.legend(prop={'size': 6})#, loc='best')
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.set_xlim(xmin=1, xmax=n_pds)
        self.axes.set_ylim(ymin=0, ymax=1.1 * np.max(self.occupancy_full))
        self.axes.set_xlabel('PD Numbers', fontsize=6)
        self.axes.set_ylabel('Occupancy', fontsize=6)
        #self.axes.autoscale()

        self.canvas.draw()

        # threshold inputs:
        def choose_thresholds():
            self.thresh_container.thresh_low = int(self.entry_low.value())
            self.thresh_container.thresh_high = int(self.entry_high.value())
            self.confirmed = False

            self.in_thres_count = np.sum((self.occupancy_full >= self.thresh_container.thresh_low))
            if self.thresh_container.thresh_high < self.thresh_container.thresh_low:
                self.in_thres_count = 0

            self.entry_prd.setValue(self.in_thres_count)


        label_low = QLabel('Low Threshold:')
        label_high = QLabel('High Threshold:')

        self.in_thres_count = np.sum((self.occupancy_full >= self.thresh_container.thresh_low))
        self.entry_prd = QDoubleSpinBox(self)
        self.entry_prd.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_prd.setDecimals(0)
        self.entry_prd.setMaximum(10000)
        self.entry_prd.setDisabled(True)
        self.entry_prd.setValue(self.in_thres_count)

        self.button_replot = QPushButton('Update plot')
        self.button_replot.clicked.connect(self.replot)

        self.btn_update = QPushButton('Update Thresholds')
        self.btn_update.clicked.connect(self.confirmThresh)
        self.btn_update.setDefault(False)
        self.btn_update.setAutoDefault(False)

        self.entry_low = QDoubleSpinBox(self)
        self.entry_low.setDecimals(0)
        self.entry_low.setMinimum(5)
        self.entry_low.setMaximum(np.amax(self.occupancy_full))
        self.entry_low.setValue(int(params.prd_thres_low))

        self.entry_high = QDoubleSpinBox(self)
        self.entry_high.setDecimals(0)
        self.entry_high.setMinimum(90)
        self.entry_high.setMaximum(10000)
        self.entry_high.setValue(int(params.prd_thres_high))

        self.entry_low.valueChanged.connect(choose_thresholds)
        self.entry_high.valueChanged.connect(choose_thresholds)

        label_prd = QLabel('Number of PDs:')

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(self.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_low, 3, 0, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.entry_low, 3, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_high, 3, 2, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.entry_high, 3, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_prd, 3, 4, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.entry_prd, 3, 5, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.button_replot, 3, 6, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_update, 3, 7, 1, 1, QtCore.Qt.AlignVCenter)
        self.setLayout(layout)


    def replot(self):
        x = np.arange(self.xlimLo, self.xlimHi + 1)
        self.lineL.set_data(x, [self.thresh_container.thresh_low])
        self.lineH.set_data(x, [self.thresh_container.thresh_high])
        self.canvas.draw()


    def confirmThresh(self):
        print(f'Number of PDs within thresholds: {self.in_thres_count}')
        if self.in_thres_count > 2:
            params.prd_thres_low = self.thresh_container.thresh_low
            params.prd_thres_high = self.thresh_container.thresh_high
            params.save()  #send new GUI data to parameters file

            print('')
            print('New thresholds set:')
            print('high:', params.prd_thres_high)
            print('low:', params.prd_thres_low)
            print('')

            self.confirmed = True

            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Set Thresholds')
            box.setIcon(QMessageBox.Information)
            box.setText('<b>Thresholding Complete</b>')
            msg = 'New high and low thresholds have been set.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            box.exec_()


class ThreshFinalCanvas(QDialog):
    def __init__(self, parent=None):
        super(ThreshFinalCanvas, self).__init__(parent)

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(1, 1, 1, polar=True)
        self.sp = self.axes.scatter([0], [0], edgecolor='k', linewidth=.5, c=[0], s=5,
                                    cmap=colormap.jet)  #empty for init
        self.cbar = self.figure.colorbar(self.sp, pad=0.1)
        #self.axes.autoscale()

        # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
        theta_labels = ['±180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']
        self.axes.set_ylim(0, 180)
        self.axes.set_yticks(np.arange(0, 180, 20))
        self.axes.xaxis.set_major_locator(FixedLocator(self.axes.get_xticks().tolist()))
        self.axes.set_xticklabels(theta_labels)
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.axes.grid(alpha=0.2)

        self.layout = QGridLayout()
        #layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #self.layout.addWidget(self.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        self.layout.addWidget(self.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        self.setLayout(self.layout)

    def plot(self, phis, thetas, occupancies):
        self.cbar.remove()
        self.axes.clear()

        theta_labels = ['±180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']
        self.axes.set_ylim(0, 180)
        self.axes.set_yticks(np.arange(0, 180, 20))
        xticks = self.axes.get_xticks().tolist()
        self.axes.xaxis.set_major_locator(FixedLocator(xticks))
        self.axes.set_xticklabels(theta_labels)

        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)

        self.axes.grid(alpha=0.2)
        thresh = self.axes.scatter(phis,
                                   thetas,
                                   edgecolor='k',
                                   linewidth=.5,
                                   c=occupancies,
                                   s=5,
                                   cmap=colormap.jet,
                                   vmin=0.,
                                   vmax=np.amax(occupancies))
        thresh.set_alpha(0.75)
        self.cbar = self.figure.colorbar(thresh, pad=0.13)
        self.cbar.ax.tick_params(labelsize=6)
        self.cbar.ax.set_title(label='Occupancy', size=6)
        self.canvas.draw()


class OccHistCanvas(QDialog):
    def __init__(self, parent):
        super(OccHistCanvas, self).__init__(parent)
        self.thresh_container = parent

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.ticklabel_format(useOffset=False, style='plain')
        self.axes.get_xaxis().get_major_formatter().set_scientific(False)

        layout = QGridLayout()

        self.label_bins = QLabel('Histogram bins:')
        self.label_bins.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.entry_bins = QDoubleSpinBox(self)
        self.entry_bins.setDecimals(0)
        self.entry_bins.setMinimum(2)
        self.entry_bins.setValue(2)
        self.entry_bins.setMaximum(1000)
        self.entry_bins.valueChanged.connect(self.change_bins)
        self.entry_bins.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        layout.addWidget(self.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_bins, 2, 1, 1, 1)
        layout.addWidget(self.entry_bins, 2, 2, 1, 1)
        self.setLayout(layout)

    def change_bins(self, occupancies):
        numBins = int(self.entry_bins.value())

        # replot self:
        self.axes.clear()
        counts, bins, bars = self.axes.hist(occupancies, bins=numBins, align='left',\
                                            edgecolor='w', linewidth=1, color='#1f77b4') #C0

        self.axes.set_xticks(bins)  #[:-1]
        self.axes.set_title('PD Occupancy Distribution', fontsize=6)
        self.axes.set_xlabel('PD Occupancy', fontsize=5)
        self.axes.set_ylabel('Number of PDs', fontsize=5)
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.get_xaxis().get_major_formatter().set_scientific(False)
        self.axes.ticklabel_format(useOffset=False, style='plain')

        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)

        self.axes.autoscale()
        self.figure.canvas.draw()
