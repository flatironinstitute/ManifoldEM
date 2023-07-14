import imageio

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QDialog, QTabWidget, QLabel, QFrame, QPushButton, QSlider,
                             QLayout, QGridLayout, QSpinBox, QComboBox, QCheckBox)

from ManifoldEM.data_store import data_store
from ManifoldEM.params import p


class VidCanvas(QDialog):
    imgDir = ''
    img_paths = []
    imgs = []
    run = 0  #switch, {-1,0,1} :: {backwards,pause,forward}
    f = 0  #frame index (current frame)
    rec = 0  #safeguard for recursion limit
    delay = .001  #playback delay in ms


    def get_frame_count(self):
        return len(self.imgs)


    def __init__(self, gif_path: str, parent=None):
        super(VidCanvas, self).__init__(parent)

        self.imgs = list(imageio.get_reader(gif_path))

        self.figure = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        for item in [self.figure, self.ax]:
            item.patch.set_visible(False)

        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.currentIMG = self.ax.imshow(self.imgs[0], cmap='gray')  #plot initial data
        self.canvas.draw()  #refresh canvas

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
        #layout.addWidget(self.toolbar, 0,0,1,5)
        layout.addWidget(self.canvas, 2, 0, 1, 5)
        layout.addWidget(self.buttonB1, 3, 0, 1, 1)
        layout.addWidget(self.buttonBackward, 3, 1, 1, 1)
        layout.addWidget(self.buttonPause, 3, 2, 1, 1)
        layout.addWidget(self.buttonForward, 3, 3, 1, 1)
        layout.addWidget(self.buttonF1, 3, 4, 1, 1)
        layout.addWidget(self.slider, 4, 0, 1, 5)
        self.setLayout(layout)


    def scroll(self, frame):
        self.canvas.stop_event_loop()

        self.slider.setValue(self.f)
        self.currentIMG.set_data(self.imgs[self.f])  #update data
        self.canvas.draw()  #refresh canvas
        self.canvas.start_event_loop(self.delay)

        max_index = self.get_frame_count()
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
            self.canvas.stop_event_loop()


    def F1(self):  #forward one frame
        self.buttonPause.setDisabled(True)
        self.buttonForward.setDisabled(False)
        self.buttonBackward.setDisabled(False)
        self.buttonF1.setFocus()

        self.run = 0
        self.rec = 0
        self.canvas.stop_event_loop()
        if self.f == self.get_frame_count() - 1:
            self.f = 0
        else:
            self.f += 1

        self.slider.setValue(self.f)
        self.currentIMG.set_data(self.imgs[self.f])  #update data
        self.canvas.draw()  #refresh canvas


    def forward(self):  #play forward
        self.buttonForward.setDisabled(True)
        self.buttonBackward.setDisabled(False)
        self.buttonPause.setDisabled(False)
        self.buttonPause.setFocus()

        self.run = 1
        self.rec = 0
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
        self.canvas.stop_event_loop()
        if self.f == 0:
            self.f = self.get_frame_count() - 1
        else:
            self.f -= 1

        self.slider.setValue(self.f)
        self.currentIMG.set_data(self.imgs[self.f])  #update data
        self.canvas.draw()  #refresh canvas


    def sliderUpdate(self):  #update frame based on user slider position
        if self.f != self.slider.value():  #only if user moves slider position manually
            self.buttonPause.setDisabled(True)
            self.buttonForward.setDisabled(False)
            self.buttonBackward.setDisabled(False)
            self.run = 0
            self.rec = 0
            self.canvas.stop_event_loop()

            self.f = self.slider.value()
            self.currentIMG.set_data(self.imgs[self.slider.value()])  #update data
            self.canvas.draw()


class _CCDetailsView(QMainWindow):
    def __init__(self, prd_index: int, psi_index: int):
        super(_CCDetailsView, self).__init__()
        self.prd_index = prd_index
        self.psi_index = psi_index
        self.initUI()


    def initUI(self):
        # Manifold2dCanvas.eigChoice1 = 0
        # Manifold2dCanvas.eigChoice2 = 1
        # Manifold2dCanvas.eigChoice3 = 2

        gif_path = p.get_psi_gif(self.prd_index, self.psi_index)
        self.vid_tab1 = VidCanvas(gif_path, parent=self)
        # vid_tab2 = Manifold2dCanvas(self)
        # vid_tab3 = Manifold3dCanvas(self)
        # vid_tab4 = ChronosCanvas(self)
        # vid_tab5 = PsiCanvas(self)
        # vid_tab6 = TauCanvas(self)
        # global vid_tabs
        vid_tabs = QTabWidget(self)
        vid_tabs.addTab(self.vid_tab1, 'Movie Player')
        # vid_tabs.addTab(vid_tab2, '2D Embedding')
        # vid_tabs.addTab(vid_tab3, '3D Embedding')
        # vid_tabs.addTab(vid_tab4, 'Chronos')
        # vid_tabs.addTab(vid_tab5, 'Psi Analysis')
        # vid_tabs.addTab(vid_tab6, 'Tau Analysis')
        #vid_tabs.setTabEnabled(1, False)
        vid_tabs.currentChanged.connect(self.onTabChange)  #signal for tabs changed via direct click

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(vid_tabs)
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()


    def closeEvent(self, ce):  #when user clicks to exit via subwindow button
        self.vid_tab1.f = 0
        self.vid_tab1.run = 0  #needed to pause scrollbar before it is deleted
        self.vid_tab1.canvas.stop_event_loop()

        if int(0) < Manifold2dCanvas.progress1.value() < int(100):  #no escaping mid-thread
            ce.ignore()


    def onTabChange(self, i):
        if i != 0:  #needed to stop `Movie Player` if tab changed during playback
            self.vid_tab1.run = 0
            self.vid_tab1.canvas.stop_event_loop()
            self.vid_tab1.buttonForward.setDisabled(False)
            self.vid_tab1.buttonForward.setFocus()
            self.vid_tab1.buttonBackward.setDisabled(False)
            self.vid_tab1.buttonPause.setDisabled(True)
