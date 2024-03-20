from .import_tab import ImportTab
from .distribution_tab import DistributionTab
from .embedding_tab import EmbeddingTab
from .eigenvectors_tab import EigenvectorsTab
from .compilation_tab import CompilationTab
from .energetics_tab import EnergeticsTab
from ..params import p

from ManifoldEM.params import p

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QScrollArea, QDesktopWidget)

from typing import List, Union


class MainWindow(QMainWindow):
    proj_lev_to_tab: list[str] = [
        'Import', # 0
        'Distribution', # 1
        'Embedding', # 2
        'Embedding', # 3
        'Embedding', # 4
        'Eigenvectors', # 5
        'Compilation', # 6
        'Compilation', # 7
        'Energetics', # 8
        'Energetics', # 9
    ]
    tab_indices: dict[str, int] = {
        'Import': 0,
        'Distribution': 1,
        'Embedding': 2,
        'Eigenvectors': 3,
        'Compilation': 4,
        'Energetics': 5,
    }

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

        self.tabs = QTabWidget(self)
        self.tabs.addTab(ImportTab(self), 'Import')
        self.tabs.addTab(DistributionTab(self), 'Distribution')
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
        self.reload_tab_states()
        self.show()


    def reload_tab_states(self):
        self.set_tab_state(False)
        for tabi in range(p.project_level.value + 1):
            self.set_tab_state(True, self.proj_lev_to_tab[tabi])
            self.switch_tab(self.proj_lev_to_tab[tabi])


    def switch_tab(self, tab_name: str):
        index = self.tab_indices.get(tab_name, None)

        if index is not None:
            self.tabs.widget(index).activate()
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
