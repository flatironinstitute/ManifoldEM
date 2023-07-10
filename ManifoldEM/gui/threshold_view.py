import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QTabWidget, QDialog, QLabel,
                             QDoubleSpinBox, QAbstractSpinBox, QPushButton, QGridLayout, QLayout)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.ticker import MaxNLocator

from ManifoldEM.params import p
from ManifoldEM.data_store import data_store

class ThresholdView(QMainWindow):
    def __init__(self):
        super(ThresholdView, self).__init__()
        self.thresh_low = p.PDsizeThL
        self.thresh_high = p.PDsizeThH

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
        self.thresh_tab1 = ThreshAllCanvas(self)
        # thresh_tab2 = ThreshFinalCanvas(self)
        # thresh_tab3 = OccHistCanvas(self)
        global thresh_tabs
        thresh_tabs = QTabWidget(self)
        thresh_tabs.addTab(self.thresh_tab1, 'Edit Thresholds')
        # thresh_tabs.addTab(thresh_tab2, 'Thresholded PDs')
        # thresh_tabs.addTab(thresh_tab3, 'Occupancy Distribution')

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(thresh_tabs)
        self.setWindowModality(QtCore.Qt.ApplicationModal)  #freezes out parent window
        self.show()

        thresh_tabs.currentChanged.connect(self.onTabChange)  #signal for tab changed via direct click

    def onTabChange(self, i):
        if i == 1:  #signals when view switched to tab 2
            # re-threshold bins:
            temp_PrDs = []
            temp_occ = []
            temp_phi = []
            temp_theta = []

            for i in range(0, len(P1.all_PrDs)):
                if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                    temp_PrDs.append(i + 1)
                    temp_occ.append(P1.all_occ[i])
                    # subtract 180 is needed for scatter's label switch:
                    temp_phi.append((float(P1.all_phi[i]) - 180) * np.pi / 180)  #needed in Radians
                    temp_theta.append(float(P1.all_theta[i]))

            #def format_coord(x,y):
            #return 'Phi={:1.2f}, Theta={:1.2f}'.format(((x*180)/np.pi)-180,y)

            # replot ThreshFinalCanvas:
            try:
                # Crashes in the cbar remove sometimes, so only _try_ to cleanup
                ThreshFinalCanvas.axes.clear()
                #ThreshFinalCanvas.axes.format_coord = format_coord
                ThreshFinalCanvas.cbar.remove()
            except:
                pass
            # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
            theta_labels = [
                '%s180%s' % (u"\u00B1", u"\u00b0"),
                '-135%s' % (u"\u00b0"),
                '-90%s' % (u"\u00b0"),
                '-45%s' % (u"\u00b0"),
                '0%s' % (u"\u00b0"),
                '45%s' % (u"\u00b0"),
                '90%s' % (u"\u00b0"),
                '135%s' % (u"\u00b0")
            ]
            ThreshFinalCanvas.axes.set_ylim(0, 180)
            ThreshFinalCanvas.axes.set_yticks(np.arange(0, 180, 20))
            ThreshFinalCanvas.axes.set_xticklabels(theta_labels)
            for tick in ThreshFinalCanvas.axes.xaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            for tick in ThreshFinalCanvas.axes.yaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            ThreshFinalCanvas.axes.grid(alpha=0.2)
            thresh = ThreshFinalCanvas.axes.scatter(temp_phi,
                                                    temp_theta,
                                                    edgecolor='k',
                                                    linewidth=.5,
                                                    c=temp_occ,
                                                    s=5,
                                                    cmap=cm.jet,
                                                    vmin=0.,
                                                    vmax=float(np.amax(P1.all_occ)))
            thresh.set_alpha(0.75)
            ThreshFinalCanvas.cbar = ThreshFinalCanvas.figure.colorbar(thresh, pad=0.13)
            ThreshFinalCanvas.cbar.ax.tick_params(labelsize=6)
            ThreshFinalCanvas.cbar.ax.set_title(label='Occupancy', size=6)
            ThreshFinalCanvas.canvas.draw()

        if i == 2:
            OccHistCanvas.numBins = int(OccHistCanvas.entry_bins.value())
            # re-threshold bins:
            temp_PrDs = []
            temp_occ = []
            for i in range(0, len(P1.all_PrDs)):
                if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                    temp_PrDs.append(i + 1)
                    if P1.all_occ[i] > p.PDsizeThH:
                        temp_occ.append(p.PDsizeThH)
                    else:
                        temp_occ.append(P1.all_occ[i])

            OccHistCanvas.entry_bins.setValue(int(len(set(temp_occ)) / 2.))
            OccHistCanvas.entry_bins.setMaximum(len(set(temp_occ)))
            OccHistCanvas.entry_bins.setSuffix(' / %s' % len(set(temp_occ)))  #number of unique occupancies

            # replot OccHistCanvas:
            OccHistCanvas.axes.clear()
            counts, bins, bars = OccHistCanvas.axes.hist(temp_occ, bins=int(OccHistCanvas.numBins), align='right',\
                                                            edgecolor='w', linewidth=1, color='#1f77b4') #C0

            OccHistCanvas.axes.set_xticks(bins)
            OccHistCanvas.axes.set_title('PD Occupancy Distribution', fontsize=6)
            OccHistCanvas.axes.set_xlabel('PD Occupancy', fontsize=5)
            OccHistCanvas.axes.set_ylabel('Number of PDs', fontsize=5)
            OccHistCanvas.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)
            OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')

            for tick in OccHistCanvas.axes.xaxis.get_major_ticks():
                tick.label1.set_fontsize(4)
            for tick in OccHistCanvas.axes.yaxis.get_major_ticks():
                tick.label1.set_fontsize(4)

            OccHistCanvas.axes.autoscale()
            OccHistCanvas.figure.canvas.draw()

    def closeEvent(self, ce):  #safety message if user clicks to exit via window button
        if not self.thresh_tab1.confirmed:
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
                self.thresh_tab1.confirmed = 1
                self.thresh_low = p.PDsizeThL
                self.thresh_high = p.PDsizeThH
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
        n_pds = len(prds.occupancy_no_duplication)
        self.axes.bar(range(n_pds), prds.occupancy_no_duplication, align='center',
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
        self.lineL.set_data(x, [p.PDsizeThL])
        self.lineH.set_data(x, [p.PDsizeThH])

        self.axes.axvline(n_pds + 1, color='#7f7f7f', linestyle='-', linewidth=.5)

        #self.axes.legend(prop={'size': 6})#, loc='best')
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes.set_xlim(xmin=1, xmax=n_pds)
        self.axes.set_ylim(ymin=0, ymax=1.1 * np.max(prds.occupancy_no_duplication))
        self.axes.set_xlabel('PD Numbers', fontsize=6)
        self.axes.set_ylabel('Occupancy', fontsize=6)
        #self.axes.autoscale()

        self.canvas.draw()

        # threshold inputs:
        def choose_thresholds():
            self.thresh_container.thresh_low = int(self.entry_low.value())
            self.thresh_container.thresh_high = int(self.entry_high.value())
            self.confirmed = False

            prds = data_store.get_prds()
            self.in_thres_count = np.sum((prds.occupancy_no_duplication >= self.thresh_container.thresh_low) &
                                         (prds.occupancy_no_duplication <= self.thresh_container.thresh_high))

            self.entry_prd.setValue(self.in_thres_count)

            # self.replot()

        label_low = QLabel('Low Threshold:')
        label_high = QLabel('High Threshold:')

        self.in_thres_count = np.sum((prds.occupancy_no_duplication >= self.thresh_container.thresh_low) &
                                     (prds.occupancy_no_duplication <= self.thresh_container.thresh_high))
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
        self.entry_low.setMaximum(np.amax(prds.occupancy_no_duplication))
        self.entry_low.setValue(int(p.PDsizeThL))

        self.entry_high = QDoubleSpinBox(self)
        self.entry_high.setDecimals(0)
        self.entry_high.setMinimum(90)
        self.entry_high.setMaximum(10000)
        self.entry_high.setValue(int(p.PDsizeThH))

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
        # self.axes.set_ylim(ymin=0, ymax=self.thresh_high + 20)
        self.canvas.draw()


    def confirmThresh(self):
        if self.in_thres_count > 2:
            p.PDsizeThL = self.thresh_container.thresh_low
            p.PDsizeThH = self.thresh_container.thresh_high
            p.save()  #send new GUI data to parameters file
            data_store.get_prds().update()

            print('')
            print('New thresholds set:')
            print('high:', p.PDsizeThH)
            print('low:', p.PDsizeThL)
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
        ThreshFinalCanvas.figure = Figure(dpi=200)
        ThreshFinalCanvas.canvas = FigureCanvas(ThreshFinalCanvas.figure)
        #ThreshFinalCanvas.toolbar = NavigationToolbar(ThreshFinalCanvas.canvas, self)
        ThreshFinalCanvas.axes = ThreshFinalCanvas.figure.add_subplot(1, 1, 1, polar=True)
        thresh = ThreshFinalCanvas.axes.scatter([0], [0], edgecolor='k', linewidth=.5, c=[0], s=5,
                                                cmap=cm.hsv)  #empty for init
        ThreshFinalCanvas.cbar = ThreshFinalCanvas.figure.colorbar(thresh, pad=0.1)
        #ThreshFinalCanvas.axes.autoscale()

        # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
        theta_labels = [
            '%s180%s' % (u"\u00B1", u"\u00b0"),
            '-135%s' % (u"\u00b0"),
            '-90%s' % (u"\u00b0"),
            '-45%s' % (u"\u00b0"),
            '0%s' % (u"\u00b0"),
            '45%s' % (u"\u00b0"),
            '90%s' % (u"\u00b0"),
            '135%s' % (u"\u00b0")
        ]
        ThreshFinalCanvas.axes.set_ylim(0, 180)
        ThreshFinalCanvas.axes.set_yticks(np.arange(0, 180, 20))
        ThreshFinalCanvas.axes.set_xticklabels(theta_labels)
        for tick in ThreshFinalCanvas.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in ThreshFinalCanvas.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        ThreshFinalCanvas.axes.grid(alpha=0.2)

        ThreshFinalCanvas.layout = QtGui.QGridLayout()
        #layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #ThreshFinalCanvas.layout.addWidget(ThreshFinalCanvas.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        ThreshFinalCanvas.layout.addWidget(ThreshFinalCanvas.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        self.setLayout(ThreshFinalCanvas.layout)


class OccHistCanvas(QDialog):
    numBins = 100

    def __init__(self, parent=None):
        super(OccHistCanvas, self).__init__(parent)

        # create canvas and plot data:
        OccHistCanvas.figure = Figure(dpi=200)
        OccHistCanvas.canvas = FigureCanvas(OccHistCanvas.figure)
        OccHistCanvas.toolbar = NavigationToolbar(OccHistCanvas.canvas, self)
        OccHistCanvas.axes = OccHistCanvas.figure.add_subplot(1, 1, 1)
        OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')
        OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)

        layout = QtGui.QGridLayout()

        OccHistCanvas.label_bins = QtGui.QLabel('Histogram bins:')
        OccHistCanvas.label_bins.setFont(font_standard)
        OccHistCanvas.label_bins.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        OccHistCanvas.entry_bins = QtGui.QDoubleSpinBox(self)
        OccHistCanvas.entry_bins.setDecimals(0)
        OccHistCanvas.entry_bins.setMinimum(2)
        OccHistCanvas.entry_bins.setValue(int(len(set(P1.thresh_occ)) / 2.))
        OccHistCanvas.entry_bins.setMaximum(len(set(P1.thresh_occ)))
        OccHistCanvas.entry_bins.setSuffix(' / %s' % len(set(P1.thresh_occ)))  #number of unique occupancies
        OccHistCanvas.entry_bins.valueChanged.connect(self.change_bins)
        OccHistCanvas.entry_bins.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        layout.addWidget(OccHistCanvas.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(OccHistCanvas.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(OccHistCanvas.label_bins, 2, 1, 1, 1)
        layout.addWidget(OccHistCanvas.entry_bins, 2, 2, 1, 1)
        self.setLayout(layout)

    def change_bins(self):
        OccHistCanvas.numBins = int(OccHistCanvas.entry_bins.value())
        # re-threshold bins:
        temp_PrDs = []
        temp_occ = []
        for i in range(0, len(P1.all_PrDs)):
            if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                temp_PrDs.append(i + 1)
                if P1.all_occ[i] > p.PDsizeThH:
                    temp_occ.append(p.PDsizeThH)
                else:
                    temp_occ.append(P1.all_occ[i])

        # replot OccHistCanvas:
        OccHistCanvas.axes.clear()
        counts, bins, bars = OccHistCanvas.axes.hist(temp_occ, bins=int(OccHistCanvas.numBins), align='left',\
                                                     edgecolor='w', linewidth=1, color='#1f77b4') #C0

        OccHistCanvas.axes.set_xticks(bins)  #[:-1]
        OccHistCanvas.axes.set_title('PD Occupancy Distribution', fontsize=6)
        OccHistCanvas.axes.set_xlabel('PD Occupancy', fontsize=5)
        OccHistCanvas.axes.set_ylabel('Number of PDs', fontsize=5)
        OccHistCanvas.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)
        OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')

        for tick in OccHistCanvas.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in OccHistCanvas.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)

        OccHistCanvas.axes.autoscale()
        OccHistCanvas.figure.canvas.draw()
