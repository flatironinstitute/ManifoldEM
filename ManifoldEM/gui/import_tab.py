import os

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit, QFileDialog, QMessageBox, QInputDialog,
                             QDoubleSpinBox, QGridLayout, QWidget, QPushButton)

from numbers import Number
from typing import Tuple, Union
import shutil
import tempfile

from ManifoldEM.params import params, ProjectLevel
from ManifoldEM.util import get_image_width_from_stack

def choose_pixel(entry: QDoubleSpinBox):
    params.ms_pixel_size = float(entry.value())


def choose_resolution(entry: QDoubleSpinBox):
    params.ms_estimated_resolution = float(entry.value())


def choose_diameter(entry: QDoubleSpinBox):
    params.particle_diameter = float(entry.value())


def choose_aperture(entry: QDoubleSpinBox):
    params.aperture_index = int(entry.value())


def update_shannon(entry: QDoubleSpinBox):
    entry.setValue(params.sh)


def update_ang_width(entry: QDoubleSpinBox):
    entry.setValue(params.ang_width)


def choose_avg_vol(entry, parent):
    filename = QFileDialog.getOpenFileName(parent, 'Choose Data File', '', ('Data Files (*.mrc)'))[0]
    entry.setText(filename)
    params.avg_vol_file = filename


def choose_align(entry, parent):
    filename = QFileDialog.getOpenFileName(parent, 'Choose Data File', '', ('Data Files (*.star)'))[0]
    entry.setText(filename)
    params.align_param_file = filename


def choose_stack(entry, parent):
    filename = QFileDialog.getOpenFileName(parent, 'Choose Data File', '', ('Data Files (*.mrcs)'))[0]
    entry.setText(filename)
    params.img_stack_file = filename


def choose_mask(entry, parent):
    filename = QFileDialog.getOpenFileName(parent, 'Choose Data File', '', ('Data Files (*.mrc)'))[0]
    entry.setText(filename)
    params.mask_vol_file = filename


def choose_proj_name(entry, parent):
    text, ok = QInputDialog.getText(parent, 'ManifoldEM Project Name', 'Enter project name:')
    if ok:
        try:
            with tempfile.TemporaryDirectory() as dir:
                with open(os.path.join(dir, text), 'w'):
                    pass
        except OSError:
            box = QMessageBox(parent)
            box.setWindowTitle('ManifoldEM Error')
            box.setText('<b>Input Error</b>')
            box.setIcon(QMessageBox.Information)
            box.setInformativeText('Project names must be valid filenames on your OS/filesystem')
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()
            text = params.project_name
    else:
        text = params.project_name

    entry.setText(text)
    params.project_name = text


def text_field_selector(yoffset: int, field_name: str, default_val: str, button_text: str, onclick, parent):
    layout = parent.layout
    edge = QLabel('')
    edge.setMargin(0)
    edge.setLineWidth(1)
    edge.setFrameStyle(QFrame.Panel | QFrame.Sunken)
    layout.addWidget(edge, yoffset, 0, 1, 6)
    edge.show()

    edgea = QLabel('')
    edgea.setMargin(0)
    edgea.setLineWidth(1)
    edgea.setFrameStyle(QFrame.Box | QFrame.Sunken)
    layout.addWidget(edgea, yoffset, 0, 1, 1)
    edgea.show()

    label = QLabel(field_name)
    label.setMargin(20)
    label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
    layout.addWidget(label, yoffset, 0, 1, 1)
    label.show()

    entry = QLineEdit(default_val)
    entry.setDisabled(True)
    layout.addWidget(entry, yoffset, 1, 1, 4)
    entry.show()

    button = QPushButton(f'          {button_text}          ', parent)
    button.clicked.connect(lambda: onclick(entry, parent))
    layout.addWidget(button, yoffset, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    button.show()


def num_selector(offset: Tuple[int, int], field_name: str, suffix: str, default_val: Number,
                 valid_range: Union[Tuple[Number, Number], None], decimals: int, display_only: bool,
                 onchange, parent):
    layout = parent.layout
    xoffset, yoffset = 2*offset[0], offset[1]
    edge = QLabel('')
    edge.setMargin(20)
    edge.setFrameStyle(QFrame.Panel | QFrame.Sunken)
    layout.addWidget(edge, yoffset, xoffset, 1, 2)
    edge.show()

    edgea = QLabel('')
    edgea.setMargin(20)
    edgea.setLineWidth(1)
    edgea.setFrameStyle(QFrame.Box | QFrame.Sunken)
    layout.addWidget(edgea, yoffset, xoffset, 1, 1)
    edgea.show()

    label = QLabel(field_name)
    label.setMargin(20)
    label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
    layout.addWidget(label, yoffset, xoffset, 1, 1)
    label.show()

    entry = QDoubleSpinBox(parent)
    entry.setStyleSheet("QDoubleSpinBox { width : 150px }")

    if valid_range:
        entry.setMinimum(valid_range[0])
        entry.setMaximum(valid_range[1])

    entry.setDecimals(decimals)
    entry.setSuffix(suffix)
    entry.setDisabled(display_only)
    entry.setValue(default_val)

    if onchange:
        entry.valueChanged.connect(lambda: onchange(entry))
        onchange(entry)

    layout.addWidget(entry, yoffset, xoffset + 1, 1, 1, QtCore.Qt.AlignLeft)
    entry.show()
    return entry


class ImportTab(QWidget):
    def __init__(self, parent=None):
        super(ImportTab, self).__init__(parent)
        self.parent = parent

        layout = self.layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        text_field_selector(0, "Average Volume", params.avg_vol_file, "Browse", choose_avg_vol, self)
        text_field_selector(1, "Alignment File", params.align_param_file, "Browse", choose_align, self)
        text_field_selector(2, "Image Stack", params.img_stack_file, "Browse", choose_stack, self)
        text_field_selector(3, "Mask Volume", params.mask_vol_file, "Browse", choose_mask, self)
        text_field_selector(4, "Project Name", params.project_name, "Choose", choose_proj_name, self)

        pixel_selector = num_selector((0, 5), "Pixel Size", " \u00c5", params.ms_pixel_size, (0.001, 1000.0), 3, False, choose_pixel, self)
        diam_selector = num_selector((1, 5), "Object Diameter", " \u00c5", params.particle_diameter, (0.01, 10000.0), 2, False, choose_diameter, self)
        shannon_entry = num_selector((2, 5), "Shannon Angle", " rad", params.sh, None, 3, True, None, self)
        resolution_selector = num_selector((0, 6), "Resolution", " \u00c5", params.ms_estimated_resolution, (0.01, 1000.0), 2, False, choose_resolution, self)
        aperture_selector = num_selector((1, 6), "Aperture Index", "", params.aperture_index, (1, 1000), 0, False, choose_aperture, self)
        ang_width_entry = num_selector((2, 6), "Angle Width", " rad", params.ang_width, None, 3, True, None, self)

        for selector in (pixel_selector, diam_selector, resolution_selector, aperture_selector):
            selector.valueChanged.connect(lambda: update_shannon(shannon_entry))
            selector.valueChanged.connect(lambda: update_ang_width(ang_width_entry))

        update_shannon(shannon_entry)
        update_ang_width(ang_width_entry)

        self.load_button = QPushButton("Load existing project", self)
        layout.addWidget(self.load_button, 7, 2, 1, 1)
        self.load_button.clicked.connect(self.load)
        self.load_button.show()

        self.finalize_button = QPushButton("View Orientation Distribution", self)
        self.finalize_button.setToolTip("All entries must be complete.")
        layout.addWidget(self.finalize_button, 7, 3, 1, 1)
        self.finalize_button.clicked.connect(self.finalize)
        self.finalize_button.show()

        self.show()

    def load(self):
        filename = QFileDialog.getOpenFileName(self, 'Choose project config', '', ('Project configs (*.toml)'))[0]

        if filename:
            params.load(filename)
            self.parent.reload_tab_states()
            self.finalize_button.setDisabled(True)

    def finalize(self):
        self.finalize_button.setDisabled(True)

        if (params.avg_vol_file and params.img_stack_file and params.align_param_file and params.sh and params.ang_width):
            if os.path.exists(params.out_dir) or os.path.exists(f'params_{params.project_name}.toml'):
                box = QMessageBox(self)
                box.setWindowTitle("ManifoldEM Conflict")
                box.setText("<b>Directory Conflict</b>")
                box.setIcon(QMessageBox.Warning)
                box.setInformativeText("This project already exists, from either a previous project "
                                       "or from re-adjusting the hyper-parameters within the current run.\n\n"
                                       "Do you want to overwrite this data and proceed?")
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                reply = box.exec_()

                if reply == QMessageBox.Yes:
                    if os.path.exists(params.out_dir):
                        shutil.rmtree(params.out_dir)
                    if os.path.exists(f"params_{params.project_name}.toml"):
                        os.remove(f"params_{params.project_name}.toml")
                else:
                    self.finalize_button.setDisabled(False)
                    return

            params.ms_num_pixels = get_image_width_from_stack(params.img_stack_file)
            params.save()
            params.create_dir()

            params.project_level = ProjectLevel.BINNING
            self.parent.set_tab_state(True, "Distribution")
            self.parent.switch_tab("Distribution")
        else:
            box = QMessageBox(self)
            box.setWindowTitle("ManifoldEM Error")
            box.setText("<b>Input Error</b>")
            box.setIcon(QMessageBox.Information)
            box.setInformativeText("All values must be complete and nonzero.")
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()
            self.finalize_button.setDisabled(False)

    def activate(self):
        pass
