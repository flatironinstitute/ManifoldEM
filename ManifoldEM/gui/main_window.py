from .import_tab import ImportTab
from .distribution_tab import DistributionTab
from .embedding_tab import EmbeddingTab
from .eigenvectors_tab import EigenvectorsTab
from .compilation_tab import CompilationTab
from .energetics_tab import EnergeticsTab

from ManifoldEM.params import p

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QScrollArea, QDesktopWidget)

from typing import List, Union


class MainWindow(QMainWindow):
    tab_indices = {
        'Import': 0,
        'Distribution': 1,
        'Embedding': 2,
        'Eigenvectors': 3,
        'Compilation': 4,
        'Energetics': 5,
    }
    tab_names = {val: key for (key, val) in tab_indices.items()}

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('ManifoldEM')

        # Ensures cleanup in proper order
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        max_screen_size = QDesktopWidget().screenGeometry(-1)
        self.setMinimumSize(500, 300)
        w, h = max_screen_size.width(), max_screen_size.height()
        if w < h:
            h = (9 * w) // 16
        self.resize((7 * w) // 10, (7 * h) // 10)

        self.distribution_tab = DistributionTab(self)
        self.tabs = QTabWidget(self)
        self.tabs.addTab(ImportTab(self), 'Import')
        self.tabs.addTab(self.distribution_tab, 'Distribution')
        self.tabs.addTab(EmbeddingTab(self), 'Embedding')
        self.tabs.addTab(EigenvectorsTab(self), 'Eigenvectors')
        self.tabs.addTab(CompilationTab(self), 'Compilation')
        self.tabs.addTab(EnergeticsTab(self), 'Energetics')

        groupscroll = QHBoxLayout()
        groupscrollbox = QGroupBox()
        tablist = QVBoxLayout()
        scroll = QScrollArea()
        widget = QWidget(self)

        tablist.addWidget(self.tabs)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(groupscrollbox)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        groupscrollbox.setLayout(tablist)
        groupscroll.addWidget(scroll)

        self.setCentralWidget(scroll)

        self.set_tab_state(False, list(self.tab_indices.keys()))
        for tabi in range(p.resProj + 1):
            tab_name = self.tab_names[tabi]
            self.set_tab_state(True, tab_name)
            self.switch_tab(tab_name)

        self.show()


    def switch_tab(self, tab_name: str):
        index = self.tab_indices.get(tab_name, None)
        if tab_name == 'Distribution':
            self.distribution_tab.activate()
        if index is not None:
            self.tabs.setCurrentIndex(index)
        else:
            print(f"Invalid tab name: {tab_name}")


    def set_tab_state(self, state: bool, tab_names: Union[List[str], str, None] = None):
        if isinstance(tab_names, str):
            tab_names = [tab_names]

        if not tab_names:
            for index in range(self.tabs.count()):
                self.tabs.setTabEnabled(index, state)
            return

        for tab_name in tab_names:
            index = self.tab_indices.get(tab_name, None)
            if index is not None:
                self.tabs.setTabEnabled(index, state)
            else:
                print(f"Invalid tab name: {tab_name}")
