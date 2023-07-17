import multiprocessing
import threading

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QFrame, QPushButton, QGridLayout, QWidget, QSpinBox, QProgressBar)

from ManifoldEM.params import p
from ManifoldEM.FindConformationalCoord import op as FindConformationalCoord

class CompilationTab(QWidget):
    #temporary values:
    user_temperature = 25  #Celsius

    # threading:
    progress5Changed = QtCore.pyqtSignal(int)
    # progress6Changed = QtCore.Signal(int)

    ##########
    # Task 5:
    @QtCore.pyqtSlot()
    def start_task5(self):
        self.button_CC.setDisabled(True)
        self.button_CC.setText('Finding Conformational Coordinates')
        self.entry_temp.setDisabled(True)
        self.entry_proc.setDisabled(True)

        p.save()  #send new GUI data to user parameters file

        task5 = threading.Thread(target=FindConformationalCoord, args=(self.progress5Changed, ))
        task5.daemon = True
        task5.start()

    @QtCore.pyqtSlot(int)
    def on_progress5Changed(self, val):
        self.progress5.setValue(val)
        if val == 100:
            self.button_CC.setText('Conformational Coordinates Complete')
            # self.start_task6()

    # ##########
    # # Task 6:
    # @QtCore.Slot()
    # def start_task6(self):
    #     tabs.setTabEnabled(0, False)
    #     tabs.setTabEnabled(1, False)
    #     tabs.setTabEnabled(2, False)
    #     tabs.setTabEnabled(3, False)

    #     self.button_erg.setDisabled(True)
    #     self.button_erg.setText(' Computing Energy Landscape ')
    #     self.entry_temp.setDisabled(True)
    #     self.entry_proc.setDisabled(True)

    #     p.save()  #send new GUI data to user parameters file

    #     task6 = threading.Thread(target=EL1D.op, args=(self.progress6Changed, ))
    #     ''' ZULU
    #     if P3.user_dimensions == 1:
    #         task6 = threading.Thread(target=EL1D.op, args=(self.progress6Changed, ))
    #     else:
    #         task6 = threading.Thread(target=EL2D.op, args=(self.progress6Changed, ))
    #         '''
    #     task6.daemon = True
    #     task6.start()

    # @QtCore.Slot(int)
    # def on_progress6Changed(self, val):  #ZULU
    #     self.progress6.setValue(val)
    #     if val == 100:
    #         p.resProj = 8
    #         p.save()  #send new GUI data to user parameters file
    #         gc.collect()
    #         self.button_erg.setText('Energy Landscape Complete')

    #         fnameOM = f'{p.OM_file}OM'
    #         fnameEL = f'{p.OM_file}EL'
    #         P4.Occ1d = np.fromfile(fnameOM, dtype=int)
    #         P4.Erg1d = np.fromfile(fnameEL)

    #         Erg1dMain.entry_width.model().item(0).setEnabled(False)
    #         Erg1dMain.button_traj.setDisabled(False)
    #         P4.Erg1d = np.fromfile(fnameEL)

    #         Erg1dMain.plot_erg1d.update_figure()  #updates 1d landscape plot

    #         tabs.setTabEnabled(0, True)
    #         tabs.setTabEnabled(1, True)
    #         tabs.setTabEnabled(2, True)
    #         tabs.setTabEnabled(3, True)
    #         self.button_toP6.setDisabled(False)

    def __init__(self, parent=None):
        super(CompilationTab, self).__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        def choose_processors():
            p.ncpu = self.entry_proc.value()

        def choose_temperature():
            p.temperature = self.entry_temp.value()

        # forced space top:
        self.label_spaceT = QLabel("")
        self.label_spaceT.setMargin(0)
        layout.addWidget(self.label_spaceT, 0, 0, 1, 7, Qt.AlignVCenter)
        self.label_spaceT.show()

        def create_label(widget_args, title="", style = QFrame.Panel | QFrame.Sunken, alignment=None):
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
        create_label((1, 1, 1, 1), "Processors", alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_proc = QSpinBox(self)
        self.entry_proc.setMinimum(1)
        self.entry_proc.setMaximum(multiprocessing.cpu_count())
        self.entry_proc.valueChanged.connect(choose_processors)
        self.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        layout.addWidget(self.entry_proc, 1, 2, 1, 1, Qt.AlignLeft)
        self.entry_proc.setToolTip('The number of processors to use in parallel.')
        self.entry_proc.show()

        # temperature label + selector
        create_label((1, 3, 1, 2))
        create_label((1, 3, 1, 1), style=QFrame.Box | QFrame.Sunken)
        create_label((1, 3, 1, 1), "Temperature",
                     style=QFrame.Box | QFrame.Sunken,
                     alignment=Qt.AlignCenter | Qt.AlignVCenter)

        self.entry_temp = QSpinBox(self)
        self.entry_temp.setMinimum(0)
        self.entry_temp.setMaximum(100)
        self.entry_temp.setValue(25)
        self.entry_temp.setSuffix(' C')
        self.entry_temp.valueChanged.connect(choose_temperature)
        self.entry_temp.setStyleSheet("QSpinBox { width : 100px }")
        self.entry_temp.setToolTip('The temperature of the sample prior to quenching.')
        layout.addWidget(self.entry_temp, 1, 4, 1, 1, Qt.AlignLeft)
        self.entry_temp.show()

        # conformational coordinates progress:
        self.button_CC = QPushButton('Find Conformational Coordinates', self)
        self.button_CC.clicked.connect(self.start_task5)
        layout.addWidget(self.button_CC, 3, 1, 1, 2)
        self.button_CC.setDisabled(False)
        self.button_CC.show()

        self.progress5 = QProgressBar(minimum=0, maximum=100, value=0)
        self.progress5Changed.connect(self.on_progress5Changed)
        layout.addWidget(self.progress5, 3, 3, 1, 4)
        self.progress5.show()

        # energy landscape progress:
        self.label_Hline1 = QLabel('')
        self.label_Hline1.setMargin(0)
        self.label_Hline1.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        layout.addWidget(self.label_Hline1, 4, 1, 1, 6, Qt.AlignVCenter)
        self.label_Hline1.show()

        self.button_erg = QPushButton('Energy Landscape', self)
#        self.button_erg.clicked.connect(self.start_task6)
        layout.addWidget(self.button_erg, 5, 1, 1, 2)
        self.button_erg.setDisabled(True)
        self.button_erg.show()

        self.progress6 = QProgressBar(minimum=0, maximum=100, value=0)
#        self.progress6Changed.connect(self.on_progress6Changed)
        layout.addWidget(self.progress6, 5, 3, 1, 4)
        self.progress6.show()

        # 3d trajectories progress:
        self.label_Hline2 = QLabel('')
        self.label_Hline2.setMargin(0)
        self.label_Hline2.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        layout.addWidget(self.label_Hline2, 6, 1, 1, 6, Qt.AlignVCenter)
        self.label_Hline2.show()

        self.button_toP6 = QPushButton('View Energy Landscape', self)
#        self.button_toP6.clicked.connect(gotoP6)
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
        self.entry_proc.setValue(p.ncpu)
