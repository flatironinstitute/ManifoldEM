import imageio
import os

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QDialog, QLabel, QFrame, QPushButton, QSlider,
                             QLayout, QGridLayout, QSpinBox, QComboBox, QCheckBox)

from ManifoldEM.data_store import data_store
from ManifoldEM.params import p


class _Vid2Canvas(QDialog):
    imgs1 = []
    imgs2 = []
    run = 0  #switch, {-1,0,1} :: {backwards,pause,forward}
    f = 0  #frame index (current frame)
    rec = 0  #safeguard for recursion limit
    delay = .001  #playback delay in ms
    blank = []

    def __init__(self, parent=None):
        super(_Vid2Canvas, self).__init__(parent)
        # =====================================================================
        # Create blank image for initiation:
        # =====================================================================
        picDir = os.path.join(p.out_dir, 'topos', 'PrD_1', 'topos_1.png')
        picImg = Image.open(picDir)
        picSize = picImg.size
        self.blank = np.ones([picSize[0], picSize[1], 3], dtype=int) * 255  #white background
        # =====================================================================

        gif_path1 = os.path.join(p.out_dir, 'topos', 'PrD_1', 'psi_1.gif')
        imgs1 = imageio.get_reader(gif_path1)
        for _ in range(len(imgs1)):
            self.imgs1.append(self.blank)
            self.imgs2.append(self.blank)

        self.label_Vline = QLabel('')  #separating line
        self.label_Vline.setFrameStyle(QFrame.VLine | QFrame.Sunken)

        self.label_mov1 = QLabel('NLSA Movie 1')
        self.label_mov1.setMargin(15)
        self.label_mov1.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)


        n_prds = data_store.get_prds().n_thresholded
        self.PrD1 = QSpinBox(self)
        self.PrD1.setMinimum(1)
        self.PrD1.setMaximum(n_prds)
        self.PrD1.setPrefix('PD: ')
        self.PrD1.setDisabled(False)

        self.Psi1 = QSpinBox(self)
        self.Psi1.setMinimum(1)
        self.Psi1.setMaximum(p.num_psi)
        self.Psi1.setPrefix('Psi: ')
        self.Psi1.setDisabled(False)

        self.sense1 = QComboBox(self)
        self.sense1.addItem('Sense: FWD')
        self.sense1.addItem('Sense: REV')
        self.sense1.setToolTip('Sense for selected movie.')
        self.sense1.setDisabled(False)

        self.btnSet1 = QCheckBox('Set Movie 1')
        self.btnSet1.clicked.connect(self.setMovie1)
        self.btnSet1.setChecked(False)
        self.btnSet1.setDisabled(False)

        self.label_mov2 = QLabel('NLSA Movie 2')
        self.label_mov2.setMargin(15)
        self.label_mov2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.PrD2 = QSpinBox(self)
        self.PrD2.setMinimum(1)
        self.PrD2.setMaximum(n_prds)
        self.PrD2.setPrefix('PD: ')
        self.PrD2.setDisabled(False)

        self.Psi2 = QSpinBox(self)
        self.Psi2.setMinimum(1)
        self.Psi2.setMaximum(p.num_psi)
        self.Psi2.setPrefix('Psi: ')
        self.Psi2.setDisabled(False)

        self.sense2 = QComboBox(self)
        self.sense2.addItem('Sense: FWD')
        self.sense2.addItem('Sense: REV')
        self.sense2.setToolTip('Sense for selected movie.')
        self.sense2.setDisabled(False)

        self.btnSet2 = QCheckBox('Set Movie 2')
        self.btnSet2.clicked.connect(self.setMovie2)
        self.btnSet2.setChecked(False)
        self.btnSet2.setDisabled(False)

        self.figure1 = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax1 = self.figure1.add_axes([0, 0, 1, 1])
        self.ax1.axis('off')
        self.ax1.xaxis.set_visible(False)
        self.ax1.yaxis.set_visible(False)

        for item in [self.figure1, self.ax1]:
            item.patch.set_visible(False)

        self.canvas1 = FigureCanvas(self.figure1)
        self.currentIMG1 = self.ax1.imshow(self.imgs1[0], cmap='gray')  #plot initial data
        self.canvas1.draw()  #refresh canvas

        self.figure2 = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax2 = self.figure2.add_axes([0, 0, 1, 1])
        self.ax2.axis('off')
        self.ax2.xaxis.set_visible(False)
        self.ax2.yaxis.set_visible(False)

        for item in [self.figure2, self.ax2]:
            item.patch.set_visible(False)

        self.canvas2 = FigureCanvas(self.figure2)
        self.currentIMG2 = self.ax2.imshow(self.imgs2[0], cmap='gray')  #plot initial data
        self.canvas2.draw()  #refresh canvas

        # player control buttons:
        self.buttonF1 = QPushButton(u'\u21E5')
        self.buttonF1.clicked.connect(self.F1)
        self.buttonF1.setDisabled(False)
        self.buttonF1.setDefault(False)
        self.buttonF1.setAutoDefault(False)

        self.buttonForward = QPushButton(u'\u25B6')
        self.buttonForward.clicked.connect(self.forward)
        self.buttonForward.setDisabled(False)
        self.buttonForward.setDefault(True)
        self.buttonForward.setAutoDefault(True)

        self.buttonPause = QPushButton(u'\u25FC')
        self.buttonPause.clicked.connect(self.pause)
        self.buttonPause.setDisabled(True)
        self.buttonPause.setDefault(False)
        self.buttonPause.setAutoDefault(False)

        self.buttonBackward = QPushButton(u'\u25C0')
        self.buttonBackward.clicked.connect(self.backward)
        self.buttonBackward.setDisabled(False)
        self.buttonBackward.setDefault(False)
        self.buttonBackward.setAutoDefault(False)

        self.buttonB1 = QPushButton(u'\u21E4')
        self.buttonB1.clicked.connect(self.B1)
        self.buttonB1.setDisabled(False)
        self.buttonB1.setDefault(False)
        self.buttonB1.setAutoDefault(False)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.get_frame_count() - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.sliderUpdate)

        # create layout:
        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        layout.addWidget(self.label_mov1, 0, 2, 1, 2)
        layout.addWidget(self.label_mov2, 0, 7, 1, 2)

        layout.addWidget(self.PrD1, 1, 1, 1, 1)
        layout.addWidget(self.Psi1, 1, 2, 1, 1)
        layout.addWidget(self.sense1, 1, 3, 1, 1)
        layout.addWidget(self.btnSet1, 1, 4, 1, 1)

        layout.addWidget(self.PrD2, 1, 6, 1, 1)
        layout.addWidget(self.Psi2, 1, 7, 1, 1)
        layout.addWidget(self.sense2, 1, 8, 1, 1)
        layout.addWidget(self.btnSet2, 1, 9, 1, 1)

        layout.addWidget(self.canvas1, 2, 1, 1, 4)
        layout.addWidget(self.canvas2, 2, 6, 1, 4)

        layout.addWidget(self.buttonB1, 5, 2, 1, 1)
        layout.addWidget(self.buttonBackward, 5, 3, 1, 1)
        layout.addWidget(self.buttonPause, 5, 4, 1, 3)
        layout.addWidget(self.buttonForward, 5, 7, 1, 1)
        layout.addWidget(self.buttonF1, 5, 8, 1, 1)
        layout.addWidget(self.slider, 6, 2, 1, 7)

        layout.addWidget(self.label_Vline, 0, 5, 5, 1)

        self.setLayout(layout)


    def setMovie1(self):
        self.f = 0
        self.run = 0
        self.rec = 0
        self.canvas1.stop_event_loop()
        self.canvas2.stop_event_loop()
        self.buttonForward.setDisabled(False)
        self.buttonForward.setFocus()
        self.buttonBackward.setDisabled(False)
        self.buttonPause.setDisabled(True)

        prD = self.PrD1.value()
        psi = self.Psi1.value()
        self.gif_path1 = os.path.join(p.out_dir, 'topos', f'PrD_{prD}', f'psi_{psi}.gif')
        self.imgs1 = list(imageio.get_reader(self.gif_path1))

        if self.btnSet1.isChecked():
            self.PrD1.setDisabled(True)
            self.Psi1.setDisabled(True)
            self.sense1.setDisabled(True)

            if self.sense1.currentText() == 'Sense: REV':
                self.imgs1.reverse()
        else:
            self.PrD1.setDisabled(False)
            self.Psi1.setDisabled(False)
            self.sense1.setDisabled(False)
            self.imgs1 = [self.blank] * self.get_frame_count()

        self.canvas1.flush_events()

        self.slider.setMaximum(self.get_frame_count() - 1)
        self.currentIMG1 = self.ax1.imshow(self.imgs1[0], cmap='gray')  #plot initial frame
        self.slider.setValue(0)
        self.f = self.slider.value()

        self.canvas1.draw()  #refresh canvas 1


    def setMovie2(self):
        self.f = 0
        self.run = 0
        self.rec = 0
        self.canvas1.stop_event_loop()
        self.canvas2.stop_event_loop()
        self.buttonForward.setDisabled(False)
        self.buttonForward.setFocus()
        self.buttonBackward.setDisabled(False)
        self.buttonPause.setDisabled(True)

        prD = self.PrD2.value()
        psi = self.Psi2.value()
        self.gif_path2 = os.path.join(p.out_dir, 'topos', f'PrD_{prD}', f'psi_{psi}.gif')
        self.imgs2 = list(imageio.get_reader(self.gif_path2))

        if self.btnSet2.isChecked():
            self.PrD2.setDisabled(True)
            self.Psi2.setDisabled(True)
            self.sense2.setDisabled(True)

            if self.sense2.currentText() == 'Sense: REV':
                self.imgs2.reverse()

        else:
            self.PrD2.setDisabled(False)
            self.Psi2.setDisabled(False)
            self.sense2.setDisabled(False)
            self.imgs2 = [self.blank] * self.get_frame_count()

        self.canvas2.flush_events()

        self.slider.setMaximum(self.get_frame_count() - 1)
        self.currentIMG2 = self.ax2.imshow(self.imgs2[0], cmap='gray')  #plot initial frame
        self.slider.setValue(0)
        self.f = self.slider.value()

        self.canvas2.draw()  #refresh canvas 2


    def get_frame_count(self):
        return min(len(self.imgs1), len(self.imgs2))


    def scroll(self, frame):
        self.canvas1.stop_event_loop()
        self.canvas2.stop_event_loop()

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f])  #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f])  #update data 2
        self.canvas1.draw()  #refresh canvas 1
        self.canvas2.draw()  #refresh canvas 2
        self.canvas1.start_event_loop(self.delay)
        self.canvas2.start_event_loop(self.delay)

        max_index = self.get_frame_count() - 1
        if self.run == 1:
            if self.f < max_index:
                self.f += 1
                self.scroll(self.f)
            elif self.f == max_index:
                self.f = 0
                self.rec += 1  #recursion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.f)
        elif self.run == -1:
            if self.f > 0:
                self.f -= 1
                self.scroll(self.f)
            elif self.f == 0:
                self.f = max_index
                self.rec += 1  #recusion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.f)
        elif self.run == 0:
            self.canvas1.stop_event_loop()
            self.canvas2.stop_event_loop()


    def F1(self):  #forward one frame
        self.buttonPause.setDisabled(True)
        self.buttonForward.setDisabled(False)
        self.buttonBackward.setDisabled(False)
        self.buttonF1.setFocus()

        self.run = 0
        self.rec = 0
        self.canvas1.stop_event_loop()
        self.canvas2.stop_event_loop()
        if self.f == self.get_frame_count() - 1:
            self.f = 0
        else:
            self.f += 1

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f])  #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f])  #update data 2
        self.canvas1.draw()  #refresh canvas 1
        self.canvas2.draw()  #refresh canvas 2


    def forward(self):  #play forward
        self.buttonForward.setDisabled(True)
        self.buttonBackward.setDisabled(False)
        self.buttonPause.setDisabled(False)
        self.buttonPause.setFocus()

        self.run = 1
        self.rec = 0
        self.f = (self.f + 1) % self.get_frame_count()
        self.scroll(self.f)


    def pause(self):  #stop play
        self.buttonForward.setDisabled(False)
        self.buttonBackward.setDisabled(False)
        self.buttonPause.setDisabled(True)
        self.buttonForward.setFocus()

        self.run = 0
        self.rec = 0
        self.scroll(self.f)


    def backward(self):  #play backward
        self.buttonBackward.setDisabled(True)
        self.buttonForward.setDisabled(False)
        self.buttonPause.setDisabled(False)
        self.buttonPause.setFocus()

        self.run = -1
        self.rec = 0
        self.scroll(self.f)


    def B1(self):  #backward one frame
        self.buttonPause.setDisabled(True)
        self.buttonForward.setDisabled(False)
        self.buttonBackward.setDisabled(False)
        self.buttonB1.setFocus()

        self.run = 0
        self.rec = 0
        self.canvas1.stop_event_loop()
        self.canvas2.stop_event_loop()
        if self.f == 0:
            self.f = self.get_frame_count() - 1
        else:
            self.f -= 1

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f])  #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f])  #update data 2
        self.canvas1.draw()  #refresh canvas 1
        self.canvas2.draw()  #refresh canvas 2


    def sliderUpdate(self):  #update frame based on user slider position
        if self.f != self.slider.value():  #only if user moves slider position manually
            self.buttonPause.setDisabled(True)
            self.buttonForward.setDisabled(False)
            self.buttonBackward.setDisabled(False)
            self.run = 0
            self.rec = 0
            self.canvas1.stop_event_loop()
            self.canvas2.stop_event_loop()

            self.f = self.slider.value()
            self.currentIMG1.set_data(self.imgs1[self.slider.value()])  #update data 1
            self.currentIMG2.set_data(self.imgs2[self.slider.value()])  #update data 2
            self.canvas1.draw()  #refresh canvas 1
            self.canvas2.draw()
