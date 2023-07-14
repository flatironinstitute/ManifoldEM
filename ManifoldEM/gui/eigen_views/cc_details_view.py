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

        self.prd_index = prd_index

        self.figure = Figure(dpi=200)
        self.ax = self.figure.add_subplot(111)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)

        psi_file = p.get_psi_file(prd_index - 1)  #current embedding
        data = myio.fin1(psi_file)
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
        if len(self.ax.lines) != 0:
            if self.connected == 0:
                for i in range(1, (len(self.coordsX) * 2)):
                    del (self.ax.lines[-1])  #delete all vertices and edges

            elif self.connected == 1:
                for i in range(1, (len(self.coordsX) * 2) + 1):
                    del (self.ax.lines[-1])  #delete all vertices and edges
                self.connected = 0
        else:
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
        if len(self.coordsX) > 2:
            ax = self.figure.axes[0]
            ax.plot([self.coordsX[0], self.coordsX[-1]], [self.coordsY[0], self.coordsY[-1]],
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
        self.pts_new = []
        self.pts_newX = []
        self.pts_newY = []

        codes = []
        for i in range(len(self.coordsX)):
            if i == 0:
                codes.extend([pltPath.Path.MOVETO])
            elif i == len(self.coordsX):
                codes.extend([pltPath.Path.CLOSEPOLY])
            else:
                codes.extend([pltPath.Path.LINETO])

        path = pltPath.Path(list(map(list, zip(self.coordsX, self.coordsY))), codes)
        inside = path.contains_points(np.dstack((self.pts_origX, self.pts_origY))[0].tolist(),
                                      radius=1e-9)

        sums = 0  #number of points within polygon
        index = 0
        for i in inside:
            index += 1
            if i == False:
                self.pts_newX.append(self.pts_origX[index - 1])
                self.pts_newY.append(self.pts_origY[index - 1])
                self.pts_new = zip(self.pts_newX, self.pts_newY)
            else:
                sums += 1  #number of encircled points


        # crop out points, redraw and resize figure:
        self.ax.clear()
        self.ax.scatter(self.pts_newX, self.pts_newY, s=1, c='#1f77b4')
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
        box.setFont(font_standard)
        box.setIcon(QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = box.exec_()
        if reply == QMessageBox.Yes:
            self.btn_reset.setDisabled(True)
            self.btn_rebed.setDisabled(True)
            self.parent.vid_tabs.setTabEnabled(0, False)
            self.parent.vid_tabs.setTabEnabled(2, False)
            self.parent.vid_tabs.setTabEnabled(3, False)
            self.parent.vid_tabs.setTabEnabled(4, False)
            self.parent.vid_tabs.setTabEnabled(5, False)

            prds = data_store.get_prds()
            if self.prd_index - 1 not in prds.reembed_ids:  #only make a copy of current if this is user's first re-embedding
                backup.op(self.prd_index, 1)  #makes copy in Topos/PrD and DiffMaps
                prds.reembed_ids.add(self.prd_index - 1)
                
            self.pts_orig, pts_orig_zip = itertools.tee(self.pts_orig)
            self.pts_new, pts_new_zip = itertools.tee(self.pts_new)

            embedd.op(list(pts_orig_zip), list(pts_new_zip), self.prd_index - 1)  #updates all manifold files for PD

            self.start_task1()
        else:
            pass


    def revert(self):
        msg = "Performing this action will revert the manifold for the \
                current PD back to its original embedding.\
                <br /><br />\
                Do you want to proceed?"

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM')
        box.setText('<b>Revert Manifold</b>')
        box.setFont(font_standard)
        box.setIcon(QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if box.exec_() == QMessageBox.No:
            return

        self.btn_reset.setDisabled(False)
        self.btn_rebed.setDisabled(True)
        self.btn_revert.setDisabled(True)

        self.parent.vid_tabs.setTabEnabled(0, False)
        self.parent.vid_tabs.setTabEnabled(2, False)
        self.parent.vid_tabs.setTabEnabled(3, False)
        self.parent.vid_tabs.setTabEnabled(4, False)
        self.parent.vid_tabs.setTabEnabled(5, False)

        prds = data_store.get_prds()
        prds.reembed_ids.discard(self.prd_index - 1)
        backup.op(self.prd_index, -1)

        psi_file = p.get_psi_file(self.prd_index - 1)  #current embedding
        data = myio.fin1(psi_file)
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

        self.parent.vid_tabs.setTabEnabled(0, True)
        self.parent.vid_tabs.setTabEnabled(2, True)
        self.parent.vid_tabs.setTabEnabled(3, True)
        self.parent.vid_tabs.setTabEnabled(4, True)
        self.parent.vid_tabs.setTabEnabled(5, True)

        msg = f'The manifold for PD {self.prd_index} has been successfully reverted.'
        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Reversion')
        box.setText('<b>Revert Manifold</b>')
        box.setIcon(QMessageBox.Warning)
        box.setFont(font_standard)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Ok)
        box.setDefaultButton(QMessageBox.Ok)
        box.exec_()

        # force-update main GUI window (topos images)
        self.data_changed.emit()


    def view(self):  #view average of all images in encircled region
        self.pts_encircled = []
        self.pts_encircledX = []
        self.pts_encircledY = []

        codes = []
        for i in range(len(self.coordsX)):
            if i == 0:
                codes.extend([pltPath.Path.MOVETO])
            elif i == len(self.coordsX) - 1:
                codes.extend([pltPath.Path.CLOSEPOLY])
            else:
                codes.extend([pltPath.Path.LINETO])

        path = pltPath.Path(list(map(list, zip(self.coordsX, self.coordsY))), codes)
        inside = path.contains_points(np.dstack((self.pts_origX, self.pts_origY))[0].tolist(),
                                      radius=1e-9)

        idx_encircled = []
        index = 0
        index_enc = 0
        for i in inside:
            index += 1
            if i == True:
                index_enc += 1
                self.pts_encircledX.append(self.pts_origX[index - 1])
                self.pts_encircledY.append(self.pts_origY[index - 1])
                self.pts_encircled = zip(self.pts_encircledX, self.pts_encircledY)
                idx_encircled.append(index - 1)

        print('Encircled Points:', index_enc)

        self.imgAvg = clusterAvg(idx_encircled, self.prd_index - 1)
        self.ClusterAvg()


    def ClusterAvg(self):
        global ClusterAvgMain_window
        try:
            ClusterAvgMain_window.close()
        except:
            pass
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        ClusterAvgMain_window = ClusterAvgMain()
        ClusterAvgMain_window.setMinimumSize(10, 10)
        ClusterAvgMain_window.setWindowTitle(f'Projection Direction {self.prd_index}')
        ClusterAvgMain_window.show()


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
            self.parent.vid_tabs.setTabEnabled(0, True)
            self.parent.vid_tabs.setTabEnabled(2, True)
            self.parent.vid_tabs.setTabEnabled(3, True)
            self.parent.vid_tabs.setTabEnabled(4, True)
            self.parent.vid_tabs.setTabEnabled(5, True)

            msg = f'The manifold for PD {self.prd_index} has been successfully re-embedded.'
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Re-embedding')
            box.setText('<b>Re-embed Manifold</b>')
            box.setIcon(QMessageBox.Warning)
            box.setFont(font_standard)
            box.setInformativeText(msg)
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            ret = box.exec_()

            # force-update main GUI window (topos images)
            self.data_changed.emit()



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
        # self.vid_tab2 = Manifold2dCanvas(self.prd_index, self)
        # vid_tab3 = Manifold3dCanvas(self)
        # vid_tab4 = ChronosCanvas(self)
        # vid_tab5 = PsiCanvas(self)
        # vid_tab6 = TauCanvas(self)
        self.vid_tabs = QTabWidget(self)
        self.vid_tabs.addTab(self.vid_tab1, 'Movie Player')
        # self.vid_tabs.addTab(vid_tab2, '2D Embedding')
        # vid_tabs.addTab(vid_tab3, '3D Embedding')
        # vid_tabs.addTab(vid_tab4, 'Chronos')
        # vid_tabs.addTab(vid_tab5, 'Psi Analysis')
        # vid_tabs.addTab(vid_tab6, 'Tau Analysis')
        #vid_tabs.setTabEnabled(1, False)
        self.vid_tabs.currentChanged.connect(self.onTabChange)  #signal for tabs changed via direct click

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(self.vid_tabs)
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()


    def closeEvent(self, ce):  #when user clicks to exit via subwindow button
        self.vid_tab1.frame_id = 0
        self.vid_tab1.run = 0  #needed to pause scrollbar before it is deleted
        self.vid_tab1.canvas.stop_event_loop()

        if int(0) < Manifold2dCanvas.progress1.value() < int(100):  #no escaping mid-thread
            ce.ignore()


    def onTabChange(self, i):
        if i != 0:  #needed to stop `Movie Player` if tab changed during playback
            self.vid_tab1.run = 0
            self.vid_tab1.canvas.stop_event_loop()
            self.vid_tab1.button_play.setDisabled(False)
            self.vid_tab1.button_play.setFocus()
            self.vid_tab1.button_play_backward.setDisabled(False)
            self.vid_tab1.button_pause.setDisabled(True)


    def connect_signals(self, data_change_callback):
        return
        self.vid_tab2.data_changed.connect(data_change_callback)
