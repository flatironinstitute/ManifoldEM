import multiprocessing
import os
import threading

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit, QPushButton,
                             QGridLayout, QWidget, QSpinBox, QProgressBar)

from ManifoldEM.params import p
from ManifoldEM.util import remote_runner, is_valid_host


class EmbeddingTab(QWidget):
    distance_progress_changed = QtCore.pyqtSignal(int)
    eigenvector_progress_changed = QtCore.pyqtSignal(int)
    psi_progress_changed = QtCore.pyqtSignal(int)
    nlsa_progress_changed = QtCore.pyqtSignal(int)
    hostname = ""

    @QtCore.pyqtSlot()
    def calc_distances(self):
        if self.hostname and not is_valid_host(self.hostname):
            print(f"Invalid hostname: {self.hostname}")
            return

        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_hostname.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_dist.setDisabled(True)
        self.button_dist.setText('Distance Calculation Initiated')

        if self.hostname:
            cmd = f'manifold-cli -n {p.ncpu} calc-distance --num-psis {p.num_psis}'
            task = threading.Thread(target=remote_runner,
                                    args=(self.hostname, cmd, self.distance_progress_changed))
        else:
            from ManifoldEM.GetDistancesS2 import op as GetDistancesS2
            task = threading.Thread(target=GetDistancesS2, args=(None, self.distance_progress_changed, ))

        task.daemon = True
        task.start()
    
    @QtCore.pyqtSlot(int)
    def on_distance_progress_changed(self, val):
        self.distance_progress.setValue(val)
        if val == self.distance_progress.maximum():
            self.button_dist.setText('Distance Calculation Complete')
            self.button_eig.setDisabled(False)
            p.save()
            self.calc_eigenvectors()

    @QtCore.pyqtSlot()
    def calc_eigenvectors(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_hostname.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_eig.setDisabled(True)
        self.button_eig.setText('Embedding Initiated')

        if self.hostname:
            cmd = f'manifold-cli -n {p.ncpu} manifold-analysis'
            task = threading.Thread(target=remote_runner,
                                    args=(self.hostname, cmd, self.eigenvector_progress_changed))
        else:
            from ManifoldEM.manifoldAnalysis import op as calculate_eigenvectors
            task = threading.Thread(target=calculate_eigenvectors, args=(self.eigenvector_progress_changed, ))
        
        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_eigenvector_progress_changed(self, val):
        self.eigenvector_progress.setValue(val)
        if val == self.eigenvector_progress.maximum():
            self.button_eig.setText('Embedding Complete')
            self.button_psi.setDisabled(False)
            p.save()
            self.calc_psi()

    @QtCore.pyqtSlot()
    def calc_psi(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_hostname.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_psi.setDisabled(True)
        self.button_psi.setText('Spectral Analysis Initiated')

        if self.hostname:
            cmd = f'manifold-cli -n {p.ncpu} psi-analysis'
            task = threading.Thread(target=remote_runner,
                                    args=(self.hostname, cmd, self.psi_progress_changed))
        else:
            from ManifoldEM.psiAnalysis import op as psi_analysis
            task = threading.Thread(target=psi_analysis, args=(self.psi_progress_changed, ))

        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_psi_progress_changed(self, val):
        self.psi_progress.setValue(val)
        if val == self.psi_progress.maximum():
            self.button_psi.setText('Spectral Analysis Complete')
            p.save()
            self.calc_nlsa()

    @QtCore.pyqtSlot()
    def calc_nlsa(self):
        p.save()

        self.entry_proc.setDisabled(True)
        self.entry_hostname.setDisabled(True)
        self.entry_psi.setDisabled(True)
        self.entry_dim.setDisabled(True)
        self.button_nlsa.setDisabled(True)
        self.button_nlsa.setText('NLSA Movie Initiated')

        if self.hostname:
            cmd = f'manifold-cli -n {p.ncpu} nlsa-movie'
            task = threading.Thread(target=remote_runner,
                                    args=(self.hostname, cmd, self.nlsa_progress_changed))
        else:
            from ManifoldEM.NLSAmovie import op as nlsa_movie
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

        def choose_hostname():
            self.hostname = self.entry_hostname.text()

        def create_label(widget_args, title="", alignment=None):
            label = QLabel(title)
            label.setMargin(20)
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

        # nproc label + selector
        create_label((1, 1, 1, 1), "Processors", alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_proc = QSpinBox(self)
        self.entry_proc.setMinimum(1)
        self.entry_proc.setMaximum(256)
        self.entry_proc.setValue(p.ncpu)
        self.entry_proc.valueChanged.connect(choose_processors)
        self.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        self.entry_proc.setToolTip('The number of processors to use in parallel.')
        layout.addWidget(self.entry_proc, 1, 2, 1, 1, Qt.AlignLeft)
        self.entry_proc.show()


        # hostname selector
        create_label((2, 1, 1, 1), "Hostname", alignment=Qt.AlignCenter | Qt.AlignVCenter)
        self.entry_hostname = QLineEdit(self)
        self.entry_hostname.textChanged.connect(choose_hostname)
        layout.addWidget(self.entry_hostname, 2, 2, 1, 1, Qt.AlignLeft)
        self.entry_hostname.show()

        # psi label + selector
        create_label((1, 3, 1, 1),
                     title="Eigenvectors",
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
        create_label((1, 5, 1, 1),
                     title="Dimensions",
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

        row = 3
        create_hline((row, 1, 1, 6))
        row += 1
        
        # distances progress:
        self.button_dist = QPushButton('Distance Calculation', self)
        self.button_dist.clicked.connect(self.calc_distances)
        layout.addWidget(self.button_dist, row, 1, 1, 2)
        self.button_dist.setDisabled(False)
        self.button_dist.show()

        self.distance_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.distance_progress_changed.connect(self.on_distance_progress_changed)
        layout.addWidget(self.distance_progress, row, 3, 1, 4)
        self.distance_progress.show()

        # eigenvectors progress:
        row += 1
        create_hline((row, 1, 1, 6))
        row += 1

        self.button_eig = QPushButton('Embedding', self)
        self.button_eig.clicked.connect(self.calc_eigenvectors)
        layout.addWidget(self.button_eig, row, 1, 1, 2)
        self.button_eig.setDisabled(True)
        self.button_eig.show()

        self.eigenvector_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.eigenvector_progress_changed.connect(self.on_eigenvector_progress_changed)
        layout.addWidget(self.eigenvector_progress, row, 3, 1, 4)
        self.eigenvector_progress.show()

        # spectral anaylsis progress:
        row += 1
        create_hline((row, 1, 1, 6))
        row += 1

        self.button_psi = QPushButton('Spectral Analysis', self)
        self.button_psi.clicked.connect(self.calc_psi)
        layout.addWidget(self.button_psi, row, 1, 1, 2)
        self.button_psi.setDisabled(True)
        self.button_psi.show()

        self.psi_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.psi_progress_changed.connect(self.on_psi_progress_changed)
        layout.addWidget(self.psi_progress, row, 3, 1, 4)
        self.psi_progress.show()

        # nlsa movie progress:
        row += 1
        create_hline((row, 1, 1, 6))
        row += 1

        self.button_nlsa = QPushButton("NLSA Movie", self)
        self.button_nlsa.clicked.connect(self.calc_nlsa)
        layout.addWidget(self.button_nlsa, row, 1, 1, 2)
        self.button_nlsa.setDisabled(True)
        self.button_nlsa.show()

        self.nlsa_progress = QProgressBar(minimum=0, maximum=100, value=0)
        self.nlsa_progress_changed.connect(self.on_nlsa_progress_changed)
        layout.addWidget(self.nlsa_progress, row, 3, 1, 4)
        self.nlsa_progress.show()

        row += 1
        create_hline((row, 1, 1, 6))
        row += 1

        self.button_to_eigenvectors = QPushButton('View Eigenvectors', self)
        self.button_to_eigenvectors.clicked.connect(self.finalize)
        layout.addWidget(self.button_to_eigenvectors, row, 3, 1, 2)
        self.button_to_eigenvectors.setDisabled(True)
        self.button_to_eigenvectors.show()

        # extend spacing:
        self.label_space = QLabel("")
        self.label_space.setMargin(0)
        layout.addWidget(self.label_space, row, 0, 5, 4, Qt.AlignVCenter)
        self.label_space.show()

        self.show()


    def finalize(self):
        p.resProj = 3
        p.save()
        self.main_window.set_tab_state(True, "Eigenvectors")
        self.main_window.switch_tab("Eigenvectors")


    def activate(self):
        self.entry_proc.setValue(p.ncpu)
