import multiprocessing
import threading

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QMessageBox,
                             QInputDialog, QDoubleSpinBox, QGridLayout, QWidget, QSpinBox, QProgressBar)

from numbers import Number
from typing import Tuple, Union

import numpy as np
from ManifoldEM.params import p
from ManifoldEM.GetDistancesS2 import op as GetDistancesS2
from ManifoldEM.manifoldAnalysis import op as calculate_eigenvectors
from ManifoldEM.psiAnalysis import op as psi_analysis
from ManifoldEM.NLSAmovie import op as nlsa_movie

class EmbeddingTab(QWidget):
    distance_progress_changed = QtCore.pyqtSignal(int)
    eigenvector_progress_changed = QtCore.pyqtSignal(int)
    psi_progress_changed = QtCore.pyqtSignal(int)
    nlsa_progress_changed = QtCore.pyqtSignal(int)

    @QtCore.pyqtSlot()
    def calc_distances(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_dist.setDisabled(True)
        self.button_dist.setText('Distance Calculation Initiated')
        task = threading.Thread(target=GetDistancesS2, args=(self.distance_progress_changed, ))
        task.daemon = True
        task.start()
    
    @QtCore.pyqtSlot(int)
    def on_distance_progress_changed(self, val):
        self.distance_progress.setValue(val)
        if val == self.distance_progress.maximum():
            self.button_dist.setText('Distance Calculation Complete')
            self.button_eig.setDisabled(False)
            p.resProj = 2
            p.save()
            self.calc_eigenvectors()

    @QtCore.pyqtSlot()
    def calc_eigenvectors(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_eig.setDisabled(True)
        self.button_eig.setText('Embedding Initiated')
        task = threading.Thread(target=calculate_eigenvectors, args=(self.eigenvector_progress_changed, ))
        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_eigenvector_progress_changed(self, val):
        self.eigenvector_progress.setValue(val)
        if val == self.eigenvector_progress.maximum():
            self.button_eig.setText('Embedding Complete')
            self.button_psi.setDisabled(False)
            p.resProj = 3
            p.save()
            self.calc_psi()

    @QtCore.pyqtSlot()
    def calc_psi(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_psi.setDisabled(True)
        self.button_psi.setText('Spectral Analysis Initiated')
        task = threading.Thread(target=psi_analysis, args=(self.psi_progress_changed, ))
        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_psi_progress_changed(self, val):
        self.psi_progress.setValue(val)
        if val == self.psi_progress.maximum():
            self.button_psi.setText('Spectral Analysis Complete')
            # self.button_next.setDisabled(False)
            p.resProj = 4
            p.save()
            self.calc_nlsa()

    @QtCore.pyqtSlot()
    def calc_nlsa(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_nlsa.setDisabled(True)
        self.button_nlsa.setText('Spectral Analysis Initiated')
        task = threading.Thread(target=nlsa_movie, args=(self.nlsa_progress_changed, ))
        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_nlsa_progress_changed(self, val):
        self.nlsa_progress.setValue(val)
        if val == self.nlsa_progress.maximum():
            self.button_nlsa.setText('NLSA Movie Complete')
            self.button_to_eigenvectors.setEnabled(True)
            p.resProj = 5
            p.save()


    def __init__(self, parent=None):
        super(EmbeddingTab, self).__init__(parent)
        self.main_window = parent

        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        def choose_processors():
            p.ncpu = self.entry_proc.value()

        def choose_psi():
            p.num_psis = self.entry_psi.value()

        def choose_dimensions():
            pass

        def create_label(widget_args, title="", style = QFrame.Panel | QFrame.Sunken, alignment=None):
            label = QLabel(title)
            label.setMargin(20)
            label.setLineWidth(1)
            label.setFrameStyle(style)
            if alignment:
                label.setAlignment(alignment)
            layout.addWidget(label, *widget_args)
            label.show()
            return label

        def create_hline(pos):
            hline = QLabel("")
            hline.setMargin(0)
            hline.setFrameStyle(QFrame.HLine | QFrame.Sunken)
            layout.addWidget(hline, *pos, Qt.AlignVCenter)
            hline.show()

        # forced space top:
        self.label_spaceT = QLabel("")
        self.label_spaceT.setMargin(0)
        layout.addWidget(self.label_spaceT, 0, 0, 1, 7, Qt.AlignVCenter)
        self.label_spaceT.show()

        # main outline:
        create_label((0, 0, 13, 8))
        create_label((1, 1, 1, 2))
        create_label((1, 1, 1, 1), style=QFrame.Box | QFrame.Sunken)
        
        # nproc label + selector
        create_label((1, 1, 1, 1), "Processors", alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_proc = QSpinBox(self)
        self.entry_proc.setMinimum(1)
        self.entry_proc.setMaximum(multiprocessing.cpu_count())
        self.entry_proc.setValue(p.ncpu)
        self.entry_proc.valueChanged.connect(choose_processors)
        self.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        self.entry_proc.setToolTip('The number of processors to use in parallel.')
        layout.addWidget(self.entry_proc, 1, 2, 1, 1, Qt.AlignLeft)
        self.entry_proc.show()

        # psi label + selector
        create_label((1, 3, 1, 2))
        create_label((1, 3, 1, 1), style=QFrame.Box | QFrame.Sunken)
        create_label((1, 3, 1, 1),
                     title="Eigenvectors",
                     style=QFrame.Box | QFrame.Sunken,
                     alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_psi = QSpinBox(self)
        self.entry_psi.setMinimum(1)
        self.entry_psi.setMaximum(8)
        self.entry_psi.setValue(8)
        self.entry_psi.valueChanged.connect(choose_psi)
        self.entry_psi.setStyleSheet("QSpinBox { width : 100px }")
        self.entry_psi.setToolTip('The number of DM eigenvectors to consider for NLSA.')
        layout.addWidget(self.entry_psi, 1, 4, 1, 1, Qt.AlignLeft)
        self.entry_psi.show()

        # dimension selector
        create_label((1, 5, 1, 2))
        create_label((1, 5, 1, 1), style=QFrame.Box | QFrame.Sunken)
        create_label((1, 5, 1, 1),
                     title="Dimensions",
                     style=QFrame.Box | QFrame.Sunken,
                     alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_dim = QSpinBox(self)
        self.entry_dim.setMinimum(1)
        self.entry_dim.setMaximum(1)
        self.entry_dim.setValue(1)
        self.entry_dim.setDisabled(True)
        self.entry_dim.valueChanged.connect(choose_dimensions)
        self.entry_dim.setToolTip("The number of orthogonal conformational coordinates to compare within the energy landscape.")
        self.entry_dim.setStyleSheet("QSpinBox { width : 100px }")
        layout.addWidget(self.entry_dim, 1, 6, 1, 1, Qt.AlignLeft)
        self.entry_dim.show()

        
        # distances progress:
        self.button_dist = QPushButton('Distance Calculation', self)
        self.button_dist.clicked.connect(self.calc_distances)
        layout.addWidget(self.button_dist, 3, 1, 1, 2)
        self.button_dist.setDisabled(False)
        self.button_dist.show()

        self.distance_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.distance_progress_changed.connect(self.on_distance_progress_changed)
        layout.addWidget(self.distance_progress, 3, 3, 1, 4)
        self.distance_progress.show()

        # eigenvectors progress:
        create_hline((4, 1, 1, 6))
        self.button_eig = QPushButton('Embedding', self)
        self.button_eig.clicked.connect(self.calc_eigenvectors)
        layout.addWidget(self.button_eig, 5, 1, 1, 2)
        self.button_eig.setDisabled(True)
        self.button_eig.show()

        self.eigenvector_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.eigenvector_progress_changed.connect(self.on_eigenvector_progress_changed)
        layout.addWidget(self.eigenvector_progress, 5, 3, 1, 4)
        self.eigenvector_progress.show()

        # spectral anaylsis progress:
        create_hline((6, 1, 1, 6))

        self.button_psi = QPushButton('Spectral Analysis', self)
        self.button_psi.clicked.connect(self.calc_psi)
        layout.addWidget(self.button_psi, 7, 1, 1, 2)
        self.button_psi.setDisabled(True)
        self.button_psi.show()

        self.psi_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.psi_progress_changed.connect(self.on_psi_progress_changed)
        layout.addWidget(self.psi_progress, 7, 3, 1, 4)
        self.psi_progress.show()

        # nlsa movie progress:
        create_hline((8, 1, 1, 6))

        self.button_nlsa = QPushButton("NLSA Movie", self)
        self.button_nlsa.clicked.connect(self.calc_nlsa)
        layout.addWidget(self.button_nlsa, 9, 1, 1, 2)
        self.button_nlsa.setDisabled(True)
        self.button_nlsa.show()

        self.nlsa_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.nlsa_progress_changed.connect(self.on_nlsa_progress_changed)
        layout.addWidget(self.nlsa_progress, 9, 3, 1, 4)
        self.nlsa_progress.show()

        create_hline((10, 1, 1, 6))

        self.button_to_eigenvectors = QPushButton('View Eigenvectors', self)
        self.button_to_eigenvectors.clicked.connect(self.finalize)
        layout.addWidget(self.button_to_eigenvectors, 11, 3, 1, 2)
        self.button_to_eigenvectors.setDisabled(True)
        self.button_to_eigenvectors.show()

        # extend spacing:
        self.label_space = QLabel("")
        self.label_space.setMargin(0)
        layout.addWidget(self.label_space, 11, 0, 5, 4, Qt.AlignVCenter)
        self.label_space.show()

        self.show()


    def finalize(self):
        p.save()
        self.main_window.set_tab_state(True, "Eigenvectors")
        self.main_window.switch_tab("Eigenvectors")


    def activate(self):
        self.entry_proc.setValue(p.ncpu)
