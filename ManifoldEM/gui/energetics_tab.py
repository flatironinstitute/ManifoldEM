import os
import shutil
import threading

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QMessageBox, QGridLayout, QDialog,
                             QLabel, QFrame, QComboBox, QPushButton, QProgressBar)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure

import numpy as np
from ManifoldEM.trajectory import op as trajectory
from ManifoldEM.params import params


class Erg1dMain(QDialog):
    reprepare = 0  #F/T: if trajectory has already been computed (0 if not)

    # threading:
    progress7Changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(Erg1dMain, self).__init__(parent)
        layout = QGridLayout(self)
        self.plot_erg1d = Erg1dCanvas(self)
        layout.addWidget(self.plot_erg1d, 1, 0, 8, 8)
        toolbar = NavigationToolbar(self.plot_erg1d, self)
        layout.addWidget(toolbar, 0, 0, 1, 8)

        self.label_edge1 = QLabel('')
        self.label_edge1.setMargin(20)
        self.label_edge1.setLineWidth(1)
        self.label_edge1.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout.addWidget(self.label_edge1, 9, 0, 1, 2)
        self.label_edge1.show()

        self.label_rep = QLabel('View Conformational Coordinate:')
        self.label_rep.setMargin(20)
        self.label_rep.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_rep, 9, 0, 1, 1)
        self.label_rep.show()

        self.chooseCC = QComboBox(self)
        self.chooseCC.addItem('CC 1')
        self.chooseCC.addItem('CC 2')
        if params.user_dimensions == 1:
            self.chooseCC.setDisabled(True)
        elif params.user_dimensions == 2:
            self.chooseCC.setDisabled(False)
        self.chooseCC.setToolTip('Switch between 1D conformational coordinates.')
        self.chooseCC.currentIndexChanged.connect(self.update_selection)
        layout.addWidget(self.chooseCC, 9, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.chooseCC.show()

        self.label_edge2 = QLabel('')
        self.label_edge2.setMargin(20)
        self.label_edge2.setLineWidth(1)
        self.label_edge2.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout.addWidget(self.label_edge2, 9, 2, 1, 2)
        self.label_edge2.show()

        self.label_distr = QLabel('View Distribution:')
        self.label_distr.setMargin(20)
        self.label_distr.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_distr, 9, 2, 1, 1)
        self.label_distr.show()

        self.erg2occ = QComboBox(self)
        self.erg2occ.addItem('Energy')
        self.erg2occ.addItem('Occupancy')
        self.erg2occ.setToolTip('Switch between energy and occupancy representations.')
        self.erg2occ.currentIndexChanged.connect(self.update_selection)
        layout.addWidget(self.erg2occ, 9, 3, 1, 1, QtCore.Qt.AlignLeft)
        self.erg2occ.show()

        self.label_edge3 = QLabel('')
        self.label_edge3.setMargin(20)
        self.label_edge3.setLineWidth(1)
        self.label_edge3.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout.addWidget(self.label_edge3, 9, 4, 1, 2)
        self.label_edge3.show()

        self.label_width = QLabel('Set Path Width:')
        self.label_width.setMargin(20)
        self.label_width.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_width, 9, 4, 1, 1)
        self.label_width.show()

        self.entry_width = QComboBox(self)
        self.entry_width.addItem('1 State')
        self.entry_width.model().item(0).setEnabled(False)
        self.entry_width.addItem('2 States')
        self.entry_width.addItem('3 States')
        self.entry_width.addItem('4 States')
        self.entry_width.addItem('5 States')
        self.entry_width.setToolTip('Change the range of neighboring states to average for final reconstruction.')
        self.entry_width.currentIndexChanged.connect(self.choose_width)
        layout.addWidget(self.entry_width, 9, 5, 1, 1, QtCore.Qt.AlignLeft)
        self.entry_width.show()

        self.label_edge4 = QLabel('')
        self.label_edge4.setMargin(20)
        self.label_edge4.setLineWidth(1)
        self.label_edge4.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout.addWidget(self.label_edge4, 9, 6, 1, 2)
        self.label_edge4.show()

        # 3d trajectories progress:
        self.button_traj = QPushButton('Compute 3D Trajectories', self)
        self.button_traj.clicked.connect(self.start_task7)
        layout.addWidget(self.button_traj, 11, 0, 1, 2)
        self.button_traj.setDisabled(False)
        self.button_traj.show()

        self.progress7 = QProgressBar(minimum=0, maximum=100, value=0)
        self.progress7Changed.connect(self.on_progress7Changed)
        layout.addWidget(self.progress7, 11, 2, 1, 6)
        self.progress7.show()


    def update_selection(self):
        plot_occupancy = self.erg2occ.currentText() == 'Occupancy'
        CC_index = 1 if self.chooseCC.currentText() == 'CC 1' else 2
        self.plot_erg1d.update_figure(plot_occupancy, CC_index)


    def choose_width(self):
        self.entry_width.model().item(0).setEnabled(True)
        self.entry_width.model().item(1).setEnabled(True)
        self.entry_width.model().item(2).setEnabled(True)
        self.entry_width.model().item(3).setEnabled(True)
        self.entry_width.model().item(4).setEnabled(True)

        if self.entry_width.currentText() == '1 State':
            self.entry_width.model().item(0).setEnabled(False)
            params.width_1D = int(1)
            params.save()
        if self.entry_width.currentText() == '2 States':
            self.entry_width.model().item(1).setEnabled(False)
            params.width_1D = int(2)
            params.save()
        if self.entry_width.currentText() == '3 States':
            self.entry_width.model().item(2).setEnabled(False)
            params.width_1D = int(3)
            params.save()
        if self.entry_width.currentText() == '4 States':
            self.entry_width.model().item(3).setEnabled(False)
            params.width_1D = int(4)
            params.save()
        if self.entry_width.currentText() == '5 States':
            self.entry_width.model().item(4).setEnabled(False)
            params.width_1D = int(5)
            params.save()


    ##########
    # Task 7:
    @QtCore.pyqtSlot()
    def start_task7(self):
        if self.reprepare == 1:  #ZULU
            # overwrite warning:
            msg = 'Final outputs have already been computed for a previous\
                    <i>Path Width</i> selection. To recompute final outputs\
                    with a new path width, previous outputs must be\
                    overwritten.\
                    <br /><br />\
                    Do you want to proceed?'

            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setText('<b>Overwrite Warning</b>')
            box.setIcon(QMessageBox.Warning)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()
            if reply == QMessageBox.No:
                pass

            self.progress7.setValue(0)
            # hard-remove pre-existing trajectory outputs:
            if os.path.isdir(params.bin_dir):
                shutil.rmtree(params.bin_dir)
                os.makedirs(params.bin_dir)

            self.button_traj.setDisabled(True)
            self.button_traj.setText('Computing 3D Trajectories')
            self.erg2occ.setDisabled(True)
            self.entry_width.setDisabled(True)

            params.save()  #send new GUI data to parameters file

            task7 = threading.Thread(target=trajectory, args=(self.progress7Changed, ))
            task7.daemon = True
            task7.start()

        else:  #if first time running trajectory
            self.progress7.setValue(0)
            self.button_traj.setDisabled(True)
            self.button_traj.setText('Computing 3D Trajectories')
            self.erg2occ.setDisabled(True)
            self.entry_width.setDisabled(True)

            params.save()  #send new GUI data to parameters file

            task7 = threading.Thread(target=trajectory, args=(self.progress7Changed, ))
            task7.daemon = True
            task7.start()


    @QtCore.pyqtSlot(int)
    def on_progress7Changed(self, val):
        self.progress7.setValue(val)
        if val == 100:
            params.save()  #send new GUI data to user parameters file

            self.reprepare = 1
            self.button_traj.setText('Recompute 3D Trajectories')
            self.button_traj.setDisabled(False)
            self.erg2occ.setDisabled(False)
            self.entry_width.setDisabled(False)


class Erg1dCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(dpi=200)
        self.axes = self.fig.add_subplot(111)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)


    def compute_initial_figure(self):
        self.fig.set_tight_layout(True)
        self.axes.set_xlabel('Conformational Coordinate 1', fontsize=6)
        self.axes.set_ylabel('Energy (kcal/mol)', fontsize=6)


    def update_figure(self, plot_occupancies=False, CC_coord=1):
        if plot_occupancies:
            LS1d = np.fromfile(f'{params.OM_file}OM', dtype=int)
        else:  # plot energies
            LS1d = np.fromfile(f'{params.OM1_file}EL')  #energy path for plot

        self.axes.clear()
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.fig.set_tight_layout(True)
        self.axes.set_xlabel(f'Conformational Coordinate {CC_coord}', fontsize=6)

        if plot_occupancies:
            self.axes.plot(np.arange(1, 51), LS1d, linewidth=1, c='#d62728')  #C2
            self.axes.set_title('1D Occupancy Map', fontsize=8)
            self.axes.set_ylabel('Occupancy', fontsize=6)
        else:
            self.axes.plot(np.arange(1, 51), LS1d, linewidth=1, c='#1f77b4')  #C0
            self.axes.set_title('1D Energy Path', fontsize=8)
            self.axes.set_ylabel('Energy (kcal/mol)', fontsize=6)

        self.axes.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.axes.autoscale()
        self.show()
        self.draw()


class EnergeticsTab(QMainWindow):
    def __init__(self, parent=None):
        super(EnergeticsTab, self).__init__(parent)

        self.erg_tab1 = Erg1dMain(self)
        erg_tab2 = QWidget(self)  # Erg2dMain(self)
        erg_tabs = QTabWidget(self)
        erg_tabs.addTab(self.erg_tab1, '1D Energy Path')
        erg_tabs.addTab(erg_tab2, '2D Energy Landscape')
        erg_tabs.setTabEnabled(1, False)
        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(erg_tabs)


    def activate(self):
        self.erg_tab1.update_selection()
