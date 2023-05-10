from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QMessageBox,
                             QInputDialog, QDoubleSpinBox, QGridLayout, QWidget)

from numbers import Number
from typing import Tuple, Union

import numpy as np
from ManifoldEM import p


class DistributionTab(QWidget):
    def __init__(self, parent=None):
        super(DistributionTab, self).__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.show()
