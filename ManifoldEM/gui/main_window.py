from .import_tab import ImportTab
from .distribution_tab import DistributionTab
from .embedding_tab import EmbeddingTab
from .eigenvectors_tab import EigenvectorsTab
from .compilation_tab import CompilationTab
from .energetics_tab import EnergeticsTab

from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QScrollArea, QDesktopWidget)

from typing import List, Union

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('ManifoldEM')

        self.tabs = QTabWidget(self)
        self.tabs.addTab(ImportTab(self), 'Import')
        self.tabs.addTab(DistributionTab(self), 'Distribution')
        self.tabs.addTab(EmbeddingTab(self), 'Embedding')
        self.tabs.addTab(EigenvectorsTab(self), 'Eigenvectors')
        self.tabs.addTab(CompilationTab(self), 'Compilation')
        self.tabs.addTab(EnergeticsTab(self), 'Energetics')

        self.groupscroll = QHBoxLayout()
        self.groupscrollbox = QGroupBox()

        self.MVB = QVBoxLayout()
        self.MVB.addWidget(self.tabs)

        scroll = QScrollArea()
        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.groupscrollbox)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.groupscrollbox.setLayout(self.MVB)
        self.groupscroll.addWidget(scroll)
        self.setCentralWidget(scroll)

        max_screen_size = QDesktopWidget().screenGeometry(-1)
        self.setMinimumSize(500, 300)
        self.setMaximumSize(max_screen_size.width(), max_screen_size.height())
        self.resize(7 * max_screen_size.width() // 10, 7 * max_screen_size.height() // 10)

        self.set_tab_state(False, ['Distribution', 'Embedding', 'Eigenvectors', 'Compilation', 'Energetics'])
        self.show()


    def set_tab_state(self, state: bool, tab_names: Union[List[str], None] = None):
        tab_indices = {
            'Import': 0,
            'Distribution': 1,
            'Embedding': 2,
            'Eigenvectors': 3,
            'Compilation': 4,
            'Energetics': 5,
        }

        if not tab_names:
            for index in range(self.tabs.count()):
                self.tabs.setTabEnabled(index, state)
            return

        for tab_name in tab_names:
            index = tab_indices.get(tab_name, None)
            if index:
                self.tabs.setTabEnabled(index, state)
            else:
                print(f"Invalid tab name: {tab_name}")
