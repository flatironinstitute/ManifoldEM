import threading

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QFrame,
    QPushButton,
    QGridLayout,
    QLineEdit,
    QWidget,
    QSpinBox,
    QProgressBar,
)

from ManifoldEM.params import params
from ManifoldEM.util import remote_runner, is_valid_host


class CompilationTab(QWidget):
    # temporary values:
    user_temperature = 25  # Celsius
    hostname = ""

    # threading:
    find_cc_progress_changed = QtCore.pyqtSignal(int)
    compute_landscape_progress_changed = QtCore.pyqtSignal(int)

    ##########
    # Task 5:
    @QtCore.pyqtSlot()
    def start_find_cc_task(self):
        if self.hostname and not is_valid_host(self.hostname):
            print(f"Invalid hostname: {self.hostname}")
            return

        self.button_CC.setDisabled(True)
        self.button_CC.setText("Finding Conformational Coordinates")
        self.entry_hostname.setDisabled(True)
        self.entry_temp.setDisabled(True)
        self.entry_proc.setDisabled(True)

        params.save()  # send new GUI data to user parameters file

        if self.hostname:
            cmd = f"manifold-cli -n {params.ncpu} find-ccs"
            task = threading.Thread(
                target=remote_runner,
                args=(self.hostname, cmd, self.find_cc_progress_changed),
            )
        else:
            from ManifoldEM.interfaces.simple import (
                find_conformational_coordinates,
            )
            from functools import partial

            target = partial(
                find_conformational_coordinates,
                blas_threads=1,
                progress_bar=self.find_cc_progress_changed,
            )
            task = threading.Thread(target=target)

        task.daemon = True
        task.start()

    @QtCore.pyqtSlot(int)
    def on_find_cc_progress_changed(self, val):
        self.progress_find_cc.setValue(val)
        if val == 100:
            self.button_CC.setText("Conformational Coordinates Complete")
            from ManifoldEM.interfaces.simple import energy_landscape

            energy_landscape()
            self.button_toP6.setEnabled(True)

    def __init__(self, parent=None):
        super(CompilationTab, self).__init__(parent)
        self.main_window = parent

        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        def choose_processors():
            params.ncpu = self.entry_proc.value()

        def choose_temperature():
            params.temperature = self.entry_temp.value()

        def choose_hostname():
            self.hostname = self.entry_hostname.text()

        # forced space top:
        self.label_spaceT = QLabel("")
        self.label_spaceT.setMargin(0)
        layout.addWidget(self.label_spaceT, 0, 0, 1, 7, Qt.AlignVCenter)
        self.label_spaceT.show()

        def create_label(
            widget_args, title="", style=QFrame.Panel | QFrame.Sunken, alignment=None
        ):
            label = QLabel(title)
            label.setMargin(20)
            label.setLineWidth(1)
            label.setFrameStyle(style)
            if alignment:
                label.setAlignment(alignment)
            layout.addWidget(label, *widget_args)
            label.show()

        # main outline:
        create_label((0, 0, 13, 8))
        # header
        create_label((1, 1, 1, 2))
        create_label((1, 1, 1, 1), style=QFrame.Box | QFrame.Sunken)

        # nproc label + selector
        create_label(
            (1, 1, 1, 1), "Processors", alignment=Qt.AlignCenter | Qt.AlignVCenter
        )

        self.entry_proc = QSpinBox(self)
        self.entry_proc.setMinimum(1)
        self.entry_proc.setMaximum(256)
        self.entry_proc.valueChanged.connect(choose_processors)
        self.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        layout.addWidget(self.entry_proc, 1, 2, 1, 1, Qt.AlignLeft)
        self.entry_proc.setToolTip("The number of processors to use in parallel.")
        self.entry_proc.show()

        # hostname selector
        create_label(
            (1, 3, 1, 1), "Hostname", alignment=Qt.AlignCenter | Qt.AlignVCenter
        )
        self.entry_hostname = QLineEdit(self)
        self.entry_hostname.textChanged.connect(choose_hostname)
        layout.addWidget(self.entry_hostname, 1, 4, 1, 1, Qt.AlignLeft)
        self.entry_hostname.show()

        # temperature label + selector
        create_label((1, 5, 1, 2))
        create_label((1, 5, 1, 1), style=QFrame.Box | QFrame.Sunken)
        create_label(
            (1, 5, 1, 1),
            "Temperature",
            style=QFrame.Box | QFrame.Sunken,
            alignment=Qt.AlignCenter | Qt.AlignVCenter,
        )

        self.entry_temp = QSpinBox(self)
        self.entry_temp.setMinimum(0)
        self.entry_temp.setMaximum(100)
        self.entry_temp.setValue(25)
        self.entry_temp.setSuffix(" C")
        self.entry_temp.valueChanged.connect(choose_temperature)
        self.entry_temp.setStyleSheet("QSpinBox { width : 100px }")
        self.entry_temp.setToolTip("The temperature of the sample prior to quenching.")
        layout.addWidget(self.entry_temp, 1, 6, 1, 1, Qt.AlignLeft)
        self.entry_temp.show()

        # conformational coordinates progress:
        self.button_CC = QPushButton("Find Conformational Coordinates", self)
        self.button_CC.clicked.connect(self.start_find_cc_task)
        layout.addWidget(self.button_CC, 3, 1, 1, 2)
        self.button_CC.setDisabled(False)
        self.button_CC.show()

        self.progress_find_cc = QProgressBar(minimum=0, maximum=100, value=0)
        self.find_cc_progress_changed.connect(self.on_find_cc_progress_changed)
        layout.addWidget(self.progress_find_cc, 3, 3, 1, 4)
        self.progress_find_cc.show()

        # 3d trajectories progress:
        self.label_Hline2 = QLabel("")
        self.label_Hline2.setMargin(0)
        self.label_Hline2.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        layout.addWidget(self.label_Hline2, 6, 1, 1, 6, Qt.AlignVCenter)
        self.label_Hline2.show()

        self.button_toP6 = QPushButton("View Energy Landscape", self)
        self.button_toP6.clicked.connect(self.finalize)
        layout.addWidget(self.button_toP6, 7, 3, 1, 2)
        self.button_toP6.setDisabled(True)
        self.button_toP6.show()

        # extend spacing:
        self.label_space = QLabel("")
        self.label_space.setMargin(0)
        layout.addWidget(self.label_space, 8, 0, 7, 4, Qt.AlignVCenter)
        self.label_space.show()

        self.show()

    def activate(self):
        self.entry_proc.setValue(params.ncpu)

    def finalize(self):
        self.main_window.set_tab_state(True, "Energetics")
        self.main_window.switch_tab("Energetics")
