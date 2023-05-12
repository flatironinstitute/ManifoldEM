from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QFrame, QPushButton, QGridLayout, QWidget

from .s2_view import S2View

class DistributionTab(QWidget):
    def __init__(self, parent=None):
        super(DistributionTab, self).__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.viz = S2View()
        self.ui_element = self.viz.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui_element, 0, 0, 1, 6)

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
        layout.addWidget(self.button_binPart, 2, 2, 1, 2)
        self.button_binPart.show()
        self.show()

    def activate(self):
        self.viz.load_data()
        self.viz.update_scene1(None)
        self.viz.update_scene2(None)
