from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QFrame, QPushButton, QGridLayout, QWidget

from ManifoldEM.params import params
from ManifoldEM.data_store import data_store
from .s2_view import S2View

class DistributionTab(QWidget):
    def __init__(self, parent):
        super(DistributionTab, self).__init__(parent)
        self.main_window = parent

        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.s2_viz = S2View()
        s2_widget = self.s2_viz.get_widget()
        layout.addWidget(s2_widget, 0, 0, 1, 6)

        def new_hline(xoffset):
            Hline = QLabel("")
            Hline.setMargin(20)
            Hline.setFrameStyle(QFrame.HLine | QFrame.Sunken)
            layout.addWidget(Hline, 2, xoffset, 1, 2, QtCore.Qt.AlignVCenter)
            Hline.show()

        new_hline(0)  # left of button
        new_hline(4)  # right of button

        self.button_binPart = QPushButton('Bin Particles')
        self.button_binPart.setToolTip('Proceed to embedding.')
        self.button_binPart.clicked.connect(self.finalize)
        layout.addWidget(self.button_binPart, 2, 2, 1, 2)
        self.button_binPart.show()
        self.show()

    def activate(self):
        self.s2_viz.load_data()
        self.s2_viz.update_scene1(None)
        self.s2_viz.update_scene2(None)

    def finalize(self):
        # not strictly necessary, but at least makes the "bin particles" text not a lie
        data_store.get_prds().update()
        self.main_window.set_tab_state(True, "Embedding")
        self.main_window.switch_tab("Embedding")
