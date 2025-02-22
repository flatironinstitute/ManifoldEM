import os
import shutil
import threading

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QMessageBox, QGridLayout, QDialog,
                             QLabel, QFrame, QComboBox, QPushButton, QProgressBar, QSpinBox)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import numpy as np
from ManifoldEM.params import params


class Erg1dMain(QDialog):
    reprepare = 0  #F/T: if trajectory has already been computed (0 if not)

    # threading:
    progress_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(Erg1dMain, self).__init__(parent)
        layout = QGridLayout(self)
        self.plot_erg1d = Erg1dCanvas(self)
        layout.addWidget(self.plot_erg1d, 1, 0, 8, 8)
        toolbar = NavigationToolbar(self.plot_erg1d, self)
        layout.addWidget(toolbar, 0, 0, 1, 8)

        def choose_processors():
            params.ncpu = self.entry_proc.value()

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

        self.label_ncpu = QLabel('Processes:')
        self.label_ncpu.setMargin(20)
        self.label_ncpu.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_ncpu, 9, 6, 1, 1)

        self.entry_proc = QSpinBox(self)
        self.entry_proc.setMinimum(1)
        self.entry_proc.setMaximum(256)
        self.entry_proc.setValue(params.ncpu)
        self.entry_proc.valueChanged.connect(choose_processors)
        self.entry_proc.setToolTip('The number of processors to use in parallel.')
        layout.addWidget(self.entry_proc, 9, 7, 1, 1, QtCore.Qt.AlignLeft)
        self.entry_proc.show()

        # 3d trajectories progress:
        self.button_traj = QPushButton('Compute 3D Trajectories', self)
        self.button_traj.clicked.connect(self.start_task)
        layout.addWidget(self.button_traj, 11, 0, 1, 2)
        self.button_traj.setDisabled(False)
        self.button_traj.show()

        self.progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.progress_changed.connect(self.on_progress_changed)
        layout.addWidget(self.progress, 11, 2, 1, 6)
        self.progress.show()


    def update_selection(self):
        CC_index = 1 if self.chooseCC.currentText() == 'CC 1' else 2
        self.plot_erg1d.update_figure(CC_index)


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


    @QtCore.pyqtSlot()
    def start_task(self):
        from ManifoldEM.trajectory import op as trajectory

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

            self.progress.setValue(0)
            # hard-remove pre-existing trajectory outputs:
            if os.path.isdir(params.bin_dir):
                shutil.rmtree(params.bin_dir)
                os.makedirs(params.bin_dir)

            self.button_traj.setDisabled(True)
            self.button_traj.setText('Computing 3D Trajectories')
            self.entry_width.setDisabled(True)

            params.save()  #send new GUI data to parameters file

            task = threading.Thread(target=trajectory, args=(self.progress_changed, ))
            task.daemon = True
            task.start()

        else:  #if first time running trajectory
            self.progress.setValue(0)
            self.button_traj.setDisabled(True)
            self.button_traj.setText('Computing 3D Trajectories')
            self.entry_width.setDisabled(True)

            params.save()  #send new GUI data to parameters file

            task = threading.Thread(target=trajectory, args=(self.progress_changed, ))
            task.daemon = True
            task.start()


    @QtCore.pyqtSlot(int)
    def on_progress_changed(self, val):
        self.progress.setValue(val)
        if val == 100:
            params.save()  #send new GUI data to user parameters file

            self.reprepare = 1
            self.button_traj.setText('Recompute 3D Trajectories')
            self.button_traj.setDisabled(False)
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
        self.axes.set_ylabel('Occupancy', fontsize=6)
        self.axes.set_title('1D Occupancy Map', fontsize=8)


    def update_figure(self, CC_coord=1):
        occupancies = np.fromfile(params.OM_file, dtype=int)

        self.axes.clear()
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        self.fig.set_tight_layout(True)
        self.axes.set_xlabel(f'Conformational Coordinate {CC_coord}', fontsize=6)

        self.axes.plot(np.arange(1, occupancies.size + 1), occupancies, linewidth=1, c='#d62728')

        self.axes.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.axes.autoscale()
        self.show()
        self.draw()


class ProbabilityTab(QMainWindow):
    def __init__(self, parent=None):
        super(ProbabilityTab, self).__init__(parent)

        self.erg_tab1 = Erg1dMain(self)
        erg_tab2 = QWidget(self)
        erg_tabs = QTabWidget(self)
        erg_tabs.addTab(self.erg_tab1, '1D Occupancy Map')
        erg_tabs.addTab(erg_tab2, '2D Occupancy Map')
        erg_tabs.setTabEnabled(1, False)
        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(erg_tabs)


    def activate(self):
        self.erg_tab1.update_selection()
