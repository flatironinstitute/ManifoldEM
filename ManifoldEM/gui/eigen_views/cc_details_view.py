import imageio
import itertools
import os
import pickle
import shutil

import numpy as np

from matplotlib.path import Path as PlotPath
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QDialog, QTabWidget, QMessageBox, QPushButton, QSlider,
                             QLayout, QGridLayout, QProgressBar, QLabel, QComboBox, QCheckBox, QFrame)

from . import ClusterAvgMain

from ManifoldEM.core import clusterAvg
from ManifoldEM.embedd import op as embedd
from ManifoldEM.data_store import data_store
from ManifoldEM.params import p

def _backup_restore(prd_index, backup=True):
    os.makedirs(os.path.join(p.out_dir, 'backup', 'topos'), exist_ok=True)
    os.makedirs(os.path.join(p.out_dir, 'backup', 'diff_maps'), exist_ok=True)
    os.makedirs(os.path.join(p.out_dir, 'backup', 'psi_analysis'), exist_ok=True)

    if backup:
        srcprefix = os.path.join(p.out_dir)
        dstprefix = os.path.join(p.out_dir, 'backup')
    else:
        srcprefix = os.path.join(p.out_dir, 'backup')
        dstprefix = os.path.join(p.out_dir)

    # topos
    srcdir = os.path.join(srcprefix, 'topos', f'PrD_{prd_index + 1}')
    dstdir = os.path.join(dstprefix, 'topos', f'PrD_{prd_index + 1}')
    shutil.copytree(srcdir, dstdir)

    # diff maps
    srcfile = os.path.join(srcprefix, 'diff_maps', f'gC_trimmed_psi_prD_{prd_index}')
    dstfile = os.path.join(dstprefix, 'diff_maps', f'gC_trimmed_psi_prD_{prd_index}')
    shutil.copy(srcfile, dstfile)

    # psi anaylsis
    srcfile = os.path.join(srcprefix, 'diff_maps', f'gC_trimmed_psi_prD_{prd_index}')
    dstfile = os.path.join(dstprefix, 'diff_maps', f'gC_trimmed_psi_prD_{prd_index}')
    shutil.copy(srcfile, dstfile)

    # psianalysis
    for psi in range(p.num_psis):
        srcfile = os.path.join(srcprefix, 'psi_analysis', f'S2_prD_{prd_index}_psi_{psi}')
        dstfile = os.path.join(dstprefix, 'psi_analysis', f'S2_prD_{prd_index}_psi_{psi}')
        shutil.copy(srcfile, dstfile)


class TauCanvas(QDialog):

    def __init__(self, prd_index:int, psi_index: int, parent=None):
        super(TauCanvas, self).__init__(parent)

        # tau from psi analsis:
        tau_fname = os.path.join(p.psi2_dir, 'S2_prD_%s_psi_%s' % (prd_index - 1, psi_index - 1))
        with open(tau_fname, 'rb') as f:
            tau_data = pickle.load(f)

        tau = tau_data['tau']
        taus_val = []
        taus_num = []

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.figure.set_tight_layout(True)

        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax1 = self.figure.add_subplot(1, 2, 1)
        self.ax2 = self.figure.add_subplot(1, 2, 2)

        idx = 0
        for i in tau:
            taus_val.append(i)
            taus_num.append(idx)
            idx += 1

        self.ax1.scatter(taus_val, taus_num, linewidths=.1, s=1, edgecolors='k', c=taus_num, cmap='jet')
        self.ax2.hist(tau, bins=p.nClass, color='#1f77b4')  #C0

        for tick in self.ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)

        self.ax1.set_xlabel('NLSA States', fontsize=5)
        self.ax1.set_ylabel('NLSA Image Indices', fontsize=5)

        self.ax1.set_xlim(xmin=0, xmax=1)
        self.ax1.set_ylim(ymin=0, ymax=np.shape(tau)[0])

        self.ax2.set_xlabel('NLSA States', fontsize=5)
        self.ax2.set_ylabel('NLSA Occupancy', fontsize=5)

        self.ax1.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.ax2.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.canvas.draw()  #refresh canvas

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(self.canvas, 1, 0, 4, 4)

        self.setLayout(layout)


class PsiCanvas(QDialog):
    def __init__(self, prd_index: int, psi_index: int, parent):
        super(PsiCanvas, self).__init__(parent)

        self.con_on = 0
        self.rec_on = 1

        # psis from psi analsis:
        psi_fname = os.path.join(p.psi2_dir, 'S2_prD_%s_psi_%s' % (prd_index - 1, psi_index - 1))
        with open(psi_fname, 'rb') as f:
            psi_data = pickle.load(f)

        # PsiC:
        self.psiC = psi_data['psiC1']
        self.psiC1 = self.psiC[:, 0]
        self.psiC2 = self.psiC[:, 1]
        self.psiC3 = self.psiC[:, 2]
        # Psirec 1:
        self.psirec = psi_data['psirec']
        self.psirec1 = self.psirec[:, 0]
        self.psirec2 = self.psirec[:, 1]
        self.psirec3 = self.psirec[:, 2]

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.mouse_init()

        self.ax.view_init(90, 90)

        # Matplotlib Default Colors:
        # ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        self.ax.scatter(self.psirec1,
                        self.psirec2,
                        self.psirec3,
                        label='psi_rec',
                        linewidths=.5,
                        edgecolors='k',
                        color='#1f77b4')  #C0
        #self.ax.scatter(self.psiC1, self.psiC2, self.psiC3, label='psi_con', linewidths=.5, edgecolors='k', c='#2ca02c') #C2
        self.ax.legend(loc='best', prop={'size': 6})

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(4)

        self.ax.tick_params(axis='x', which='major', pad=-3)
        self.ax.tick_params(axis='y', which='major', pad=-3)
        self.ax.tick_params(axis='z', which='major', pad=-3)
        self.ax.xaxis.labelpad = -8
        self.ax.yaxis.labelpad = -8
        self.ax.zaxis.labelpad = -8

        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (2), fontsize=6)
        self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (3), fontsize=6)

        self.canvas.draw()  #refresh canvas

        # canvas buttons:
        self.label_X = QLabel('X-axis:')
        self.label_X.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Y = QLabel('Y-axis:')
        self.label_Y.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Z = QLabel('Z-axis:')
        self.label_Z.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.combo_X = QComboBox(self)
        self.combo_X.setDisabled(False)

        self.combo_Y = QComboBox(self)
        self.combo_Y.setDisabled(False)

        self.combo_Z = QComboBox(self)
        self.combo_Z.setDisabled(False)

        for psi in range(p.num_psiTrunc):
            self.combo_X.addItem('Psi %s' % (int(psi + 1)))
            self.combo_Y.addItem('Psi %s' % (int(psi + 1)))
            self.combo_Z.addItem('Psi %s' % (int(psi + 1)))


        self.combo_X.setCurrentIndex(0)
        self.combo_Y.setCurrentIndex(1)
        self.combo_Z.setCurrentIndex(2)
        self.combo_X.model().item(1).setEnabled(False)
        self.combo_X.model().item(2).setEnabled(False)
        self.combo_Y.model().item(0).setEnabled(False)
        self.combo_Y.model().item(2).setEnabled(False)
        self.combo_Z.model().item(0).setEnabled(False)
        self.combo_Z.model().item(1).setEnabled(False)

        self.combo_X.currentIndexChanged.connect(self.choose_X)
        self.combo_Y.currentIndexChanged.connect(self.choose_Y)
        self.combo_Z.currentIndexChanged.connect(self.choose_Z)

        self.label_Vline = QLabel('')  #separating line
        self.label_Vline.setFrameStyle(QFrame.VLine | QFrame.Sunken)

        self.check_psiC = QCheckBox('Concatenation')
        self.check_psiC.clicked.connect(self.choose_con)
        self.check_psiC.setChecked(False)

        self.check_psiR = QCheckBox('Reconstruction')
        self.check_psiR.clicked.connect(self.choose_rec)
        self.check_psiR.setChecked(True)

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,4)
        layout.addWidget(self.canvas, 1, 0, 4, 4)

        layout.addWidget(self.label_X, 5, 0, 1, 1)
        layout.addWidget(self.label_Y, 6, 0, 1, 1)
        layout.addWidget(self.label_Z, 7, 0, 1, 1)
        layout.addWidget(self.combo_X, 5, 1, 1, 1)
        layout.addWidget(self.combo_Y, 6, 1, 1, 1)
        layout.addWidget(self.combo_Z, 7, 1, 1, 1)
        layout.addWidget(self.label_Vline, 5, 2, 3, 1)
        layout.addWidget(self.check_psiC, 5, 3, 1, 1)
        layout.addWidget(self.check_psiR, 6, 3, 1, 1)

        self.setLayout(layout)
        self.canvas.draw()


    def replot(self):
        # redraw and resize figure:
        self.ax.clear()

        if self.rec_on == 1:
            self.ax.scatter(self.psirec1,
                            self.psirec2,
                            self.psirec3,
                            label='psi_rec',
                            linewidths=.5,
                            edgecolors='k',
                            color='#1f77b4')  #C0
        if self.con_on == 1:
            self.ax.scatter(self.psiC1,
                            self.psiC2,
                            self.psiC3,
                            label='psi_con',
                            linewidths=.5,
                            edgecolors='k',
                            c='#2ca02c')  #C2

        if self.rec_on == 1 or self.con_on == 1:
            self.ax.legend(loc='best', prop={'size': 6})

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(4)

        self.ax.tick_params(axis='x', which='major', pad=-3)
        self.ax.tick_params(axis='y', which='major', pad=-3)
        self.ax.tick_params(axis='z', which='major', pad=-3)
        self.ax.xaxis.labelpad = -8
        self.ax.yaxis.labelpad = -8
        self.ax.zaxis.labelpad = -8
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_X.currentIndex()) + 1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Y.currentIndex()) + 1), fontsize=6)
        self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Z.currentIndex()) + 1), fontsize=6)

        self.canvas.draw()


    def choose_X(self):
        x = int(self.combo_X.currentIndex())

        self.psiC1 = self.psiC[:, x]
        self.psirec1 = self.psirec[:, x]

        for i in range(p.num_psiTrunc):
            self.combo_Y.model().item(i).setEnabled(True)
            self.combo_Z.model().item(i).setEnabled(True)

        self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
        self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
        self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
        self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)

        self.replot()


    def choose_Y(self):
        y = int(self.combo_Y.currentIndex())

        self.psiC2 = self.psiC[:, y]
        self.psirec2 = self.psirec[:, y]

        for i in range(p.num_psiTrunc):
            self.combo_X.model().item(i).setEnabled(True)
            self.combo_Z.model().item(i).setEnabled(True)

        self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
        self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
        self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
        self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)

        self.replot()


    def choose_Z(self):
        z = int(self.combo_Z.currentIndex())

        self.psiC3 = self.psiC[:, z]
        self.psirec3 = self.psirec[:, z]

        for i in range(p.num_psiTrunc):
            self.combo_X.model().item(i).setEnabled(True)
            self.combo_Y.model().item(i).setEnabled(True)

        self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
        self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
        self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
        self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)

        self.replot()


    def choose_con(self):
        if self.check_psiC.isChecked():
            self.con_on = 1
        else:
            self.con_on = 0
        self.replot()


    def choose_rec(self):
        if self.check_psiR.isChecked():
            self.rec_on = 1
        else:
            self.rec_on = 0
        self.replot()


class ChronosCanvas(QDialog):
    def __init__(self, prd_index: int, psi_index: int, parent):
        super(ChronosCanvas, self).__init__(parent)

        self.prd_index = prd_index
        self.psi_index = psi_index

        # chronos from psi analsis:
        chr_fname = os.path.join(p.psi2_dir, 'S2_prD_%s_psi_%s' % (self.prd_index - 1, self.psi_index - 1))
        with open(chr_fname, 'rb') as f:
            chr_data = pickle.load(f)

        chronos = chr_data['VX']

        # create canvas and plot data:
        figure = Figure(dpi=200)
        figure.set_tight_layout(True)
        canvas = FigureCanvas(figure)

        fst = 6  # title font size
        lft = -25  # lower xlim
        fsa = 4  # axis font size
        sides = ['top', 'bottom', 'left', 'right']

        for i in range(8):
            ax = figure.add_subplot(2, 4, i + 1)
            ax.plot(chronos[i], color="#1f77b4", linewidth=.5)
            ax.set_title(f'Chronos {i + 1}', fontsize=fst)
            ax.set_xlim(left=lft, right=len(chronos[i]) - lft)
            ax.set_xticks(np.arange(0, len(chronos[0]) + 1, len(chronos[0]) / 2))
            ax.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(fsa)

            ax.tick_params(direction='in', length=2, width=.25)

            for side in sides:
                ax.spines[side].set_linewidth(1)


        canvas.draw()

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(canvas, 1, 0, 4, 4)

        self.setLayout(layout)


class VidCanvas(QDialog):

    def get_frame_count(self):
        return len(self.imgs)


    def __init__(self, gif_path: str, parent=None):
        super(VidCanvas, self).__init__(parent)

        self.imgs = list(imageio.get_reader(gif_path))
        self.run = 0  #switch, {-1,0,1} :: {backwards,pause,forward}
        self.frame_id = 0  #frame index (current frame)
        self.rec = 0  #safeguard for recursion limit
        self.delay = 0.016666 # frame time in s


        self.figure = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        for item in [self.figure, self.ax]:
            item.patch.set_visible(False)

        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.current_image = self.ax.imshow(self.imgs[0], cmap='gray')  #plot initial data
        self.canvas.draw()  #refresh canvas

        # player control buttons:
        self.button_forward_one = QPushButton('⇥')
        self.button_forward_one.clicked.connect(self.forward_one)
        self.button_forward_one.setDisabled(False)
        self.button_forward_one.setDefault(False)
        self.button_forward_one.setAutoDefault(False)

        self.button_play = QPushButton('▶')
        self.button_play.clicked.connect(self.play)
        self.button_play.setDisabled(False)
        self.button_play.setDefault(True)
        self.button_play.setAutoDefault(True)

        self.button_pause = QPushButton('◼')
        self.button_pause.clicked.connect(self.pause)
        self.button_pause.setDisabled(True)
        self.button_pause.setDefault(False)
        self.button_pause.setAutoDefault(False)

        self.button_play_backward = QPushButton('◀')
        self.button_play_backward.clicked.connect(self.play_backward)
        self.button_play_backward.setDisabled(False)
        self.button_play_backward.setDefault(False)
        self.button_play_backward.setAutoDefault(False)

        self.button_backward_one = QPushButton('⇤')
        self.button_backward_one.clicked.connect(self.backward_one)
        self.button_backward_one.setDisabled(False)
        self.button_backward_one.setDefault(False)
        self.button_backward_one.setAutoDefault(False)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.get_frame_count() - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slider_update)

        # create layout:
        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(self.canvas, 2, 0, 1, 5)
        layout.addWidget(self.button_backward_one, 3, 0, 1, 1)
        layout.addWidget(self.button_play_backward, 3, 1, 1, 1)
        layout.addWidget(self.button_pause, 3, 2, 1, 1)
        layout.addWidget(self.button_play, 3, 3, 1, 1)
        layout.addWidget(self.button_forward_one, 3, 4, 1, 1)
        layout.addWidget(self.slider, 4, 0, 1, 5)
        self.setLayout(layout)


    def stop_movie(self):
        self.run = 0
        self.canvas.stop_event_loop()
        self.button_play.setDisabled(False)
        self.button_play.setFocus()
        self.button_play_backward.setDisabled(False)
        self.button_pause.setDisabled(True)


    def scroll(self, frame):
        self.canvas.stop_event_loop()
        self.slider.setValue(self.frame_id)
        self.current_image.set_data(self.imgs[self.frame_id])  #update data
        self.canvas.draw()  #refresh canvas
        self.canvas.start_event_loop(self.delay)

        max_index = self.get_frame_count() - 1
        if self.run == 1:
            if self.frame_id < max_index:
                self.frame_id += 1
                self.scroll(self.frame_id)
            elif self.frame_id == max_index:
                self.frame_id = 0
                self.rec += 1  #recursion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.frame_id)
        elif self.run == -1:
            if self.frame_id > 0:
                self.frame_id -= 1
                self.scroll(self.frame_id)
            elif self.frame_id == 0:
                self.frame_id = max_index
                self.rec += 1  #recusion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.frame_id)
        elif self.run == 0:
            self.canvas.stop_event_loop()


    def forward_one(self):  #forward one frame
        self.button_pause.setDisabled(True)
        self.button_play.setDisabled(False)
        self.button_play_backward.setDisabled(False)
        self.button_forward_one.setFocus()

        self.run = 0
        self.rec = 0
        self.canvas.stop_event_loop()
        if self.frame_id == self.get_frame_count() - 1:
            self.frame_id = 0
        else:
            self.frame_id += 1

        self.slider.setValue(self.frame_id)
        self.current_image.set_data(self.imgs[self.frame_id])  #update data
        self.canvas.draw()  #refresh canvas


    def play(self):  #play forward
        self.button_play.setDisabled(True)
        self.button_play_backward.setDisabled(False)
        self.button_pause.setDisabled(False)
        self.button_pause.setFocus()

        self.run = 1
        self.rec = 0
        self.scroll(self.frame_id)


    def pause(self):  #stop play
        self.button_play.setDisabled(False)
        self.button_play_backward.setDisabled(False)
        self.button_pause.setDisabled(True)
        self.button_play.setFocus()

        self.run = 0
        self.rec = 0
        self.scroll(self.frame_id)


    def play_backward(self):  #play backward
        self.button_play_backward.setDisabled(True)
        self.button_play.setDisabled(False)
        self.button_pause.setDisabled(False)
        self.button_pause.setFocus()

        self.run = -1
        self.rec = 0
        self.scroll(self.frame_id)


    def backward_one(self):  #backward one frame
        self.button_pause.setDisabled(True)
        self.button_play.setDisabled(False)
        self.button_play_backward.setDisabled(False)
        self.button_backward_one.setFocus()

        self.run = 0
        self.rec = 0
        self.canvas.stop_event_loop()
        if self.frame_id == 0:
            self.frame_id = self.get_frame_count() - 1
        else:
            self.frame_id -= 1

        self.slider.setValue(self.frame_id)
        self.current_image.set_data(self.imgs[self.frame_id])  #update data
        self.canvas.draw()  #refresh canvas


    def slider_update(self):  #update frame based on user slider position
        if self.frame_id != self.slider.value():  #only if user moves slider position manually
            self.button_pause.setDisabled(True)
            self.button_play.setDisabled(False)
            self.button_play_backward.setDisabled(False)
            self.run = 0
            self.rec = 0
            self.canvas.stop_event_loop()

            self.frame_id = self.slider.value()
            self.current_image.set_data(self.imgs[self.slider.value()])  #update data
            self.canvas.draw()



class Manifold2dCanvas(QDialog):
    progress1Changed = QtCore.Signal(int)
    progress2Changed = QtCore.Signal(int)
    data_changed = QtCore.pyqtSignal()

    def __init__(self, prd_index: int, parent):
        super(Manifold2dCanvas, self).__init__(parent)
        self.parent_view = parent

        self.prd_index = prd_index
        self.eigChoice1 = 0
        self.eigChoice2 = 1
        self.eigChoice3 = 2

        # for eigenvector specific plots:
        self.eig_current = 1
        self.eig_compare1 = 2
        self.eig_compare2 = 3
        self.coordsX = []  #user X coordinate picks
        self.coordsY = []  #user Y coordinate picks
        self.connected = 0  #binary: 0=unconnected, 1=connected
        self.pts_orig = []
        self.pts_origX = []
        self.pts_origY = []


        self.figure = Figure(dpi=200)
        self.ax = self.figure.add_subplot(111)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)

        psi_file = p.get_psi_file(prd_index - 1)  #current embedding
        with open(psi_file, 'rb') as f:
            data = pickle.load(f)
        x = data['psi'][:, self.eigChoice1]
        y = data['psi'][:, self.eigChoice2]

        self.pts_orig = zip(x, y)
        self.pts_origX = x
        self.pts_origY = y
        self.ax.scatter(self.pts_origX, self.pts_origY, s=1, c='#1f77b4')  # plot initial data, C0

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice1 + 1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice2 + 1), fontsize=6)
        self.ax.autoscale()
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()  #refresh canvas

        # canvas buttons:
        self.btn_reset = QPushButton('Reset Plot')
        self.btn_reset.clicked.connect(self.reset)
        self.btn_reset.setDisabled(False)
        self.btn_reset.setDefault(False)
        self.btn_reset.setAutoDefault(False)

        self.btn_connect = QPushButton('Connect Path')
        self.btn_connect.clicked.connect(self.connect)
        self.btn_connect.setDisabled(True)
        self.btn_connect.setDefault(False)
        self.btn_connect.setAutoDefault(False)

        self.btn_remove = QPushButton('Remove Cluster')
        self.btn_remove.clicked.connect(self.remove)
        self.btn_remove.setDisabled(True)
        self.btn_remove.setDefault(False)
        self.btn_remove.setAutoDefault(False)

        self.btn_rebed = QPushButton('Update Manifold', self)
        self.btn_rebed.clicked.connect(self.rebed)
        self.btn_rebed.setDisabled(True)
        self.btn_rebed.setDefault(False)
        self.btn_rebed.setAutoDefault(False)

        self.btn_revert = QPushButton('Revert Manifold', self)
        self.btn_revert.clicked.connect(self.revert)
        self.btn_revert.setDefault(False)
        self.btn_revert.setAutoDefault(False)

        # disable reversion if manifold hasn't been reembedded
        orig_embed = prd_index - 1 not in data_store.get_prds().reembed_ids
        self.btn_revert.setDisabled(orig_embed)

        self.btn_view = QPushButton('View Cluster')
        self.btn_view.clicked.connect(self.view)
        self.btn_view.setDisabled(True)
        self.btn_view.setDefault(False)
        self.btn_view.setAutoDefault(False)

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(self.canvas, 1, 0, 1, 6)
        layout.addWidget(self.btn_reset, 2, 0, 1, 1)
        layout.addWidget(self.btn_connect, 2, 1, 1, 1)
        layout.addWidget(self.btn_view, 2, 2, 1, 1)
        layout.addWidget(self.btn_remove, 2, 3, 1, 1)
        layout.addWidget(self.btn_rebed, 2, 4, 1, 1)
        layout.addWidget(self.btn_revert, 2, 5, 1, 1)

        self.progress1 = QProgressBar(minimum=0, maximum=100, value=0)
        layout.addWidget(self.progress1, 3, 0, 1, 6)
        self.progress1.show()

        self.progress1Changed.connect(self.on_progress1Changed)
        self.progress2Changed.connect(self.on_progress2Changed)

        self.setLayout(layout)


    def reset(self):
        self.connected = 0
        self.btn_connect.setDisabled(True)
        self.btn_remove.setDisabled(True)
        self.btn_view.setDisabled(True)
        self.btn_rebed.setDisabled(True)
        self.coordsX = []
        self.coordsY = []

        # redraw and resize figure:
        self.ax.clear()
        self.ax.scatter(self.pts_origX, self.pts_origY, s=1, c='#1f77b4')

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice1 + 1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice2 + 1), fontsize=6)
        self.ax.autoscale()
        self.canvas.draw()


    def connect(self):
        if len(self.coordsX) <= 2:
            print("Not enough points to connect")
            return

        self.coordsX.append(self.coordsX[0])
        self.coordsY.append(self.coordsY[0])
        ax = self.figure.axes[0]
        ax.plot([self.coordsX[-2], self.coordsX[-1]], [self.coordsY[-2], self.coordsY[-1]],
                color='#7f7f7f',
                linestyle='solid',
                linewidth=.5,
                zorder=1)  #C7
        self.canvas.draw()
        self.connected = 1
        self.btn_connect.setDisabled(True)
        self.btn_remove.setDisabled(False)
        self.btn_view.setDisabled(False)


    def remove(self):
        # reset cropped points if re-clicked:
        pts_newX = []
        pts_newY = []

        path = PlotPath(list(map(list, zip(self.coordsX, self.coordsY))), codes=None, closed=True, readonly=True)
        inside = path.contains_points(np.dstack((self.pts_origX, self.pts_origY))[0].tolist(),
                                      radius=1e-9)

        for index, i in enumerate(inside):
            if i == False:
                pts_newX.append(self.pts_origX[index])
                pts_newY.append(self.pts_origY[index])
        self.pts_new = zip(pts_newX, pts_newY)


        # crop out points, redraw and resize figure:
        self.ax.clear()
        self.ax.scatter(pts_newX, pts_newY, s=1, c='#1f77b4')
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice1 + 1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice2 + 1), fontsize=6)
        self.ax.autoscale()
        self.canvas.draw()
        self.btn_remove.setDisabled(True)
        self.btn_view.setDisabled(True)
        self.btn_rebed.setDisabled(False)


    def rebed(self):
        msg = 'Performing this action will recalculate the manifold \
                embedding step for the current PD to include only the points shown.\
                <br /><br />\
                Do you want to proceed?'

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM')
        box.setText('<b>Update Manifold</b>')
        box.setIcon(QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if box.exec_() == QMessageBox.No:
            return

        self.btn_reset.setDisabled(True)
        self.btn_rebed.setDisabled(True)
        self.parent_view.vid_tabs.setTabEnabled(0, False)
        self.parent_view.vid_tabs.setTabEnabled(2, False)
        self.parent_view.vid_tabs.setTabEnabled(3, False)
        self.parent_view.vid_tabs.setTabEnabled(4, False)
        self.parent_view.vid_tabs.setTabEnabled(5, False)

        prds = data_store.get_prds()
        if self.prd_index - 1 not in prds.reembed_ids:  #only make a copy of current if this is user's first re-embedding
            _backup_restore(self.prd_index - 1, backup=True)  #makes copy in Topos/PrD and DiffMaps
            prds.reembed_ids.add(self.prd_index - 1)

        self.pts_orig, pts_orig_zip = itertools.tee(self.pts_orig)
        self.pts_new, pts_new_zip = itertools.tee(self.pts_new)

        embedd(list(pts_orig_zip), list(pts_new_zip), self.prd_index - 1)  #updates all manifold files for PD

        self.start_task1()



    def revert(self):
        msg = "Performing this action will revert the manifold for the \
                current PD back to its original embedding.\
                <br /><br />\
                Do you want to proceed?"

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM')
        box.setText('<b>Revert Manifold</b>')
        box.setIcon(QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if box.exec_() == QMessageBox.No:
            return

        self.btn_reset.setDisabled(False)
        self.btn_rebed.setDisabled(True)
        self.btn_revert.setDisabled(True)

        self.parent_view.vid_tabs.setTabEnabled(0, False)
        self.parent_view.vid_tabs.setTabEnabled(2, False)
        self.parent_view.vid_tabs.setTabEnabled(3, False)
        self.parent_view.vid_tabs.setTabEnabled(4, False)
        self.parent_view.vid_tabs.setTabEnabled(5, False)

        prds = data_store.get_prds()
        prds.reembed_ids.discard(self.prd_index - 1)
        _backup_restore(self.prd_index - 1, backup=False)

        psi_file = p.get_psi_file(self.prd_index - 1)  #current embedding
        with open(psi_file, 'rb') as f:
            data = pickle.load(f)

        x = data['psi'][:, self.eigChoice1]
        y = data['psi'][:, self.eigChoice2]

        # redraw and resize figure:
        self.ax.clear()
        self.pts_orig = zip(x, y)
        self.pts_origX = x
        self.pts_origY = y
        for i in self.pts_orig:
            x, y = i
            self.ax.scatter(x, y, s=1, c='#1f77b4')  #plot initial data, C0

        for i in self.pts_orig:
            x, y = i
            self.ax.scatter(x, y, s=1, c='#1f77b4')  #plot initial data, C0
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice1 + 1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (self.eigChoice2 + 1), fontsize=6)
        self.ax.autoscale()
        self.canvas.draw()

        self.parent_view.vid_tabs.setTabEnabled(0, True)
        self.parent_view.vid_tabs.setTabEnabled(2, True)
        self.parent_view.vid_tabs.setTabEnabled(3, True)
        self.parent_view.vid_tabs.setTabEnabled(4, True)
        self.parent_view.vid_tabs.setTabEnabled(5, True)

        msg = f'The manifold for PD {self.prd_index} has been successfully reverted.'
        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Reversion')
        box.setText('<b>Revert Manifold</b>')
        box.setIcon(QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Ok)
        box.setDefaultButton(QMessageBox.Ok)
        box.exec_()

        # force-update main GUI window (topos images)
        self.data_changed.emit()


    def view(self):  #view average of all images in encircled region
        path = PlotPath(list(map(list, zip(self.coordsX, self.coordsY))), closed=True, codes=None, readonly=True)
        inside_mask = path.contains_points(np.dstack((self.pts_origX, self.pts_origY))[0].tolist(),
                                      radius=1e-9)

        idx_encircled = list(np.nonzero(inside_mask)[0])
        print('Encircled Points:', len(idx_encircled))

        imgAvg = clusterAvg(idx_encircled, self.prd_index - 1)

        self.cluster_avg_window = ClusterAvgMain(imgAvg)
        self.cluster_avg_window.setMinimumSize(10, 10)
        self.cluster_avg_window.setWindowTitle(f'Projection Direction {self.prd_index}')
        self.cluster_avg_window.show()


    def onclick(self, event):
        if self.connected == 0:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                self.coordsX.append(float(ix))
                self.coordsY.append(float(iy))
                ax = self.figure.axes[0]
                ax.plot(event.xdata, event.ydata, color='#d62728', marker='+', zorder=2)  #on top, C3
                if len(self.coordsX) > 1:
                    x0, y0 = self.coordsX[-2], self.coordsY[-2]
                    x1, y1 = self.coordsX[-1], self.coordsY[-1]
                    ax.plot([x0, x1], [y0, y1], color='#7f7f7f', linestyle='solid', linewidth=.5, zorder=1)  #C7
                self.canvas.draw()
            if len(self.coordsX) > 2:
                self.btn_connect.setDisabled(False)


    ##########
    # Task 1:
    @QtCore.Slot()
    def start_task1(self):
        p.save()  #send new GUI data to parameters file

        task1 = threading.Thread(target=psiAnalysis.op, args=(self.progress1Changed, ))
        task1.daemon = True
        task1.start()


    @QtCore.Slot(int)
    def on_progress1Changed(self, val):
        self.progress1.setValue(val / 2)
        if val / 2 == 50:
            self.start_task2()


    ##########
    # Task 2:
    @QtCore.Slot()
    def start_task2(self):
        p.save()  #send new GUI data to parameters file

        task2 = threading.Thread(target=NLSAmovie.op, args=(self.progress2Changed, ))
        task2.daemon = True
        task2.start()


    @QtCore.Slot(int)
    def on_progress2Changed(self, val):
        self.progress1.setValue(val / 2 + 50)
        if (val / 2 + 50) == 100:
            self.parent_view.vid_tabs.setTabEnabled(0, True)
            self.parent_view.vid_tabs.setTabEnabled(2, True)
            self.parent_view.vid_tabs.setTabEnabled(3, True)
            self.parent_view.vid_tabs.setTabEnabled(4, True)
            self.parent_view.vid_tabs.setTabEnabled(5, True)

            msg = f'The manifold for PD {self.prd_index} has been successfully re-embedded.'
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Re-embedding')
            box.setText('<b>Re-embed Manifold</b>')
            box.setIcon(QMessageBox.Warning)
            box.setFont(font_standard)
            box.setInformativeText(msg)
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()

            # force-update main GUI window (topos images)
            self.data_changed.emit()



class _CCDetailsView(QMainWindow):
    def __init__(self, prd_index: int, psi_index: int):
        super(_CCDetailsView, self).__init__()
        self.prd_index = prd_index
        self.psi_index = psi_index
        self.initUI()


    def initUI(self):
        gif_path = p.get_psi_gif(self.prd_index, self.psi_index)
        self.vid_tab1 = VidCanvas(gif_path, parent=self)
        self.vid_tab2 = QDialog(self)  # Manifold2dCanvas(self.prd_index, self)
        self.vid_tab3 = QDialog(self)  # Manifold3dCanvas(self)
        self.vid_tab4 = ChronosCanvas(self.prd_index, self.psi_index, self)
        self.vid_tab5 = PsiCanvas(self.prd_index, self.psi_index, self)
        self.vid_tab6 = TauCanvas(self.prd_index, self.psi_index, self)

        self.vid_tabs = QTabWidget(self)
        self.vid_tabs.addTab(self.vid_tab1, 'Movie Player')
        self.vid_tabs.addTab(self.vid_tab2, '2D Embedding')
        self.vid_tabs.addTab(self.vid_tab3, '3D Embedding')
        self.vid_tabs.addTab(self.vid_tab4, 'Chronos')
        self.vid_tabs.addTab(self.vid_tab5, 'Psi Analysis')
        self.vid_tabs.addTab(self.vid_tab6, 'Tau Analysis')

        self.vid_tabs.currentChanged.connect(self.onTabChange)  #signal for tabs changed via direct click

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(self.vid_tabs)
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()


    def closeEvent(self, ce):
        self.vid_tab1.stop_movie()


    def onTabChange(self, i):
        if i != 0:  #needed to stop `Movie Player` if tab changed during playback
            self.vid_tab1.stop_movie()


    # FIXME attach signals
    def connect_signals(self, data_change_callback):
        return
        self.vid_tab2.data_changed.connect(data_change_callback)