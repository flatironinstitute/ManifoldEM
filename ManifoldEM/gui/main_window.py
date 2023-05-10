from .import_tab import ImportTab
from .distribution_tab import DistributionTab
from .embedding_tab import EmbeddingTab
from .eigenvectors_tab import EigenvectorsTab
from .compilation_tab import CompilationTab
from .energetics_tab import EnergeticsTab

from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QScrollArea, QDesktopWidget)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('ManifoldEM')

        self.tabs = QTabWidget(self)
        self.tabs.resize(250, 150)
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

        self.show()
