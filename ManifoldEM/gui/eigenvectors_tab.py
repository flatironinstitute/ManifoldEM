import os
import pandas

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QMessageBox, QSpinBox,
                             QDoubleSpinBox, QGridLayout, QWidget, QSplitter, QAbstractSpinBox)
from PyQt5.QtGui import QImage, QPixmap

from PIL import Image
import numpy as np

from ManifoldEM.params import p
from ManifoldEM.data_store import data_store

from traits.api import HasTraits, Instance, on_trait_change, Str, Float, Range, Enum
from traitsui.api import View, Item, Group, HGroup, VGroup, TextEditor

from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class EigValCanvas(FigureCanvas):
    # all eigenvecs/vals:
    eig_n = []
    eig_v = []
    # user-computed vecs/vals (color blue):
    eig_n1 = []
    eig_v1 = []
    # remaining vecs/vals via [eig_n - eig_n1] (color gray):
    eig_n2 = []
    eig_v2 = []

    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)
        self.plot()

    def EigValRead(self):

        EigValCanvas.eig_n = []
        EigValCanvas.eig_v = []
        EigValCanvas.eig_n1 = []
        EigValCanvas.eig_v1 = []
        EigValCanvas.eig_n2 = []
        EigValCanvas.eig_v2 = []

        fname = os.path.join(p.out_dir, 'topos', f'PrD_{P4.user_PrD}', 'eig_spec.txt')
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
        col1 = data[0]
        col2 = data[1]
        cols = np.column_stack((col1, col2))

        for i, j in cols:
            EigValCanvas.eig_n.append(int(i))
            EigValCanvas.eig_v.append(float(j))
            if int(i) <= int(p.num_psis):
                EigValCanvas.eig_n1.append(int(i))
                EigValCanvas.eig_v1.append(float(j))
            else:
                EigValCanvas.eig_n2.append(int(i))
                EigValCanvas.eig_v2.append(float(j))
        return

    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.bar(EigValCanvas.eig_n1, EigValCanvas.eig_v1, edgecolor='none', color='#1f77b4', align='center')  #C0: blue
        ax.bar(EigValCanvas.eig_n2, EigValCanvas.eig_v2, edgecolor='none', color='#7f7f7f', align='center')  #C7: gray

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        ax.set_title('Eigenvalue Spectrum', fontsize=8)
        ax.set_xlabel(r'$\mathrm{\Psi}$', fontsize=8)
        ax.set_ylabel(r'$\mathrm{\lambda}$', fontsize=8, rotation=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(0, color='k', linestyle='-', linewidth=.25)
        ax.get_xaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.get_yaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(6)
        ax.set_xticks(EigValCanvas.eig_n)
        ax.autoscale()
        self.draw()


# conformational coordinates:
def eigSpectrum(parent):
    global EigValMain_window
    try:
        EigValMain_window.close()
    except:
        pass
    EigValMain_window = EigValMain()
    EigValMain_window.setMinimumSize(10, 10)
    EigValMain_window.setWindowTitle('Projection Direction %s' % (parent.user_prd_index))
    EigValMain_window.show()

def classAvg(parent):
    global ClassAvgMain_window
    try:
        ClassAvgMain_window.close()
    except:
        pass
    ClassAvgMain_window = ClassAvgMain()
    ClassAvgMain_window.setMinimumSize(10, 10)
    ClassAvgMain_window.setWindowTitle('Projection Direction %s' % (parent.user_prd_index))
    ClassAvgMain_window.show()


class Mayavi_Rho(HasTraits):
    """View of electrostatic potential map"""

    if False:
        def update_scene3(self):
            pass

        def update_view(self, azimuth, elevation, distance):
            pass

        def view_angles(self, dialog):
            pass

        def update_euler(self):
            pass

        pass

    scene3 = Instance(MlabSceneModel, ())
    prd_index_high = 2
    isosurface = Range(2, 9, 3, mode='enum')
    volume_alpha = Enum(1.0, .8, .6, .4, .2, 0.0)
    anchors_update = []
    trash_update = []
    phi = Str
    theta = Str
    click_on = 0
    click_on_Eul = 0

    def _phi_default(self):
        return '%s%s' % (0, u"\u00b0")

    def _theta_default(self):
        return '%s%s' % (0, u"\u00b0")

    def _S2_scale_default(self):
        return float(1)

    def get_widget(self):
        return self.edit_traits(parent=self, kind='subpanel').control

    def view_angles(self):
        zoom = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)[2]
        return zoom

    def update_view(self, azimuth, elevation, distance):
        self.scene3.mlab.view(azimuth=azimuth,
                              elevation=elevation,
                              distance=distance,
                              reset_roll=False,
                              figure=Mayavi_Rho.fig3)

    def update_euler(self, phi, theta):
        self.phi = '%s%s' % (round(phi, 2), u"\u00b0")
        self.theta = '%s%s' % (round(theta, 2), u"\u00b0")


    def __init__(self, parent):
        super(Mayavi_Rho, self).__init__()
        self.parent = parent


    @on_trait_change('volume_alpha,isosurface')
    def update_scene3(self, init=False):
        # store current camera info:
        view = self.scene3.mlab.view()
        roll = self.scene3.mlab.roll()
        Mayavi_Rho.fig3 = mlab.figure(3)
        self.scene3.background = (0.0, 0.0, 0.0)

        prds = data_store.get_prds()
        s2_positions = prds.pos_thresholded

        if init:
            mlab.clf(figure=Mayavi_Rho.fig3)

            # =================================================================
            # Volume (contour):
            # =================================================================
            import mrcfile
            with mrcfile.open(p.avg_vol_file, mode='r') as mrc:
                df_vol = mrc.data

            mirror = df_vol[..., ::-1]

            cplot = mlab.contour3d(mirror, contours=self.isosurface, color=(0.5, 0.5, 0.5), figure=Mayavi_Rho.fig3)
            cplot.actor.actor.orientation = np.array([0., -90., 0.])

            cplot.actor.actor.origin = np.array([len(df_vol) / 2, len(df_vol) / 2, len(df_vol) / 2])
            cplot.actor.actor.position = np.array([-len(df_vol) / 2, -len(df_vol) / 2, -len(df_vol) / 2])

            cplot.actor.property.backface_culling = True
            cplot.compute_normals = False
            cplot.actor.property.opacity = self.volume_alpha

            # =================================================================
            # Align-to-grid data:
            # =================================================================
            phi, theta = np.mgrid[0:np.pi:11j, 0:2 * np.pi:11j]
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(1, 1, 1))
            scale_vec = p.visualization_params['S2_scale'] * len(df_vol) / np.sqrt(2.0) * np.ones(3)

            testPlot.actor.actor.scale = scale_vec
            testPlot.actor.property.opacity = 0

            # =================================================================
            # S2 Distribution (scatter):
            # =================================================================
            splot = mlab.points3d(s2_positions[0, :],
                                  s2_positions[1, :],
                                  s2_positions[2, :],
                                  prds.cluster_ids,
                                  scale_mode='none',
                                  scale_factor=0.05,
                                  figure=Mayavi_Rho.fig3)
            splot.actor.property.backface_culling = True
            splot.actor.actor.scale = scale_vec

            # =================================================================
            # S2 Anchors (sparse scatter):
            # =================================================================
            anchor_pos = s2_positions[:, prds.anchor_ids]
            aplot = mlab.points3d(anchor_pos[0, :],
                                  anchor_pos[1, :],
                                  anchor_pos[2, :],
                                  scale_mode='none',
                                  scale_factor=0.06,
                                  figure=Mayavi_Rho.fig3)

            aplot.actor.property.backface_culling = True
            aplot.glyph.color_mode = 'no_coloring'
            aplot.actor.property.color = (1.0, 1.0, 1.0)
            aplot.actor.actor.scale = scale_vec
            aplot.actor.actor.origin = np.zeros(3)
            aplot.actor.actor.position = np.zeros(3)

            self.anchors_update = aplot.mlab_source

            # =================================================================
            # S2 Trash (sparse scatter):
            # =================================================================
            trash_pos = s2_positions[:, prds.trash_ids]
            tplot = mlab.points3d(trash_pos[0, :],
                                  trash_pos[1, :],
                                  trash_pos[2, :],
                                  scale_mode='none',
                                  scale_factor=0.06,
                                  figure=Mayavi_Rho.fig3)

            tplot.actor.property.backface_culling = True
            tplot.glyph.color_mode = 'no_coloring'
            tplot.actor.property.color = (0.0, 0.0, 0.0)
            tplot.actor.actor.scale = scale_vec
            tplot.actor.actor.origin = np.zeros(3)
            tplot.actor.actor.position = np.zeros(3)

            self.trash_update = tplot.mlab_source

        else:  #only update anchors
            trash_pos = s2_positions[:, prds.trash_ids]
            anchor_pos = s2_positions[:, prds.anchor_ids]
            self.anchors_update.reset(x=anchor_pos[:, 0], y=anchor_pos[:, 1], z=anchor_pos[:, 2])
            self.trash_update.reset(x=trash_pos[:, 0], y=trash_pos[:, 1], z=trash_pos[:, 2])

        # =====================================================================
        # reposition camera to previous:
        # =====================================================================
        mlab.view(*view)
        mlab.roll(roll)

        def press_callback(vtk_obj, event):  #left mouse down callback
            self.click_on = 1

        def release_callback(vtk_obj, event):  #left mouse release callback
            if self.click_on == 1:
                self.click_on = 0
                # =============================================================
                # magnetize to nearest prd:
                # =============================================================
                # CONVENTIONS:
                # =============================================================
                # mayavi angle 0 -> [-180, 180]: azimuth, phi, longitude
                # mayavi angle 1 -> [0, 180]: elevation/inclination, theta, latitude
                # =============================================================
                angles = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)
                phi0 = angles[0] * np.pi / 180
                theta0 = angles[1] * np.pi / 180

                r0 = np.array([
                    np.sin(theta0) * np.cos(phi0),
                    np.sin(theta0) * np.sin(phi0),
                    np.cos(theta0)
                ])

                prds = data_store.get_prds()
                dr = np.linalg.norm(prds.pos_thresholded.T - r0, axis=1)
                idx = np.argmin(dr)

                # # update view:
                phi, theta = prds.phi_thresholded[idx], prds.theta_thresholded[idx]
                self.update_euler(phi, theta)
                self.parent.entry_prd.setValue(idx + 1)  #update prd and thus topos


        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)

        # live update of Euler angles:
        def press_callback_Eul(vtk_obj, event):  #left mouse down callback
            self.click_on_Eul = 1

        def hold_callback_Eul(vtk_obj, event):  #camera rotate callback
            if self.click_on_Eul > 0:
                viewS2 = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)
                self.phi = '%s%s' % (round(viewS2[0], 2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1], 2), u"\u00b0")

        def release_callback_Eul(vtk_obj, event):  #left mouse release callback
            if self.click_on_Eul == 1:
                self.click_on_Eul = 0

        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback_Eul)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('InteractionEvent', hold_callback_Eul)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback_Eul)

    title = Str

    def _title_default(self):
        return 'Electrostatic Potential Map'

    view = View(
        VGroup(
            Group(
                Item('title',
                     springy=False,
                     show_label=False,
                     style='readonly',
                     style_sheet='*{font: "Arial"; font-size:12px; qproperty-alignment:AlignCenter}'),
                Item('scene3',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=1,
                     width=1,
                     show_label=False,
                     springy=True),
            ),
            VGroup(
                HGroup(
                    Item(
                        'phi',
                        springy=True,
                        show_label=True,  #style='readonly',
                        editor=TextEditor(evaluate=float),
                        enabled_when='phi == float(0)',  #i.e., never
                    ),
                    Item(
                        'theta',
                        springy=True,
                        show_label=True,  #style='readonly',
                        editor=TextEditor(evaluate=float),
                        enabled_when='phi == float(0)',  #i.e., never
                    ),
                ),
                show_border=False,
                orientation='vertical'),
        ),
        resizable=True,
    )


class EigenvectorsTab(QWidget):
    def __init__(self, parent=None):
        super(EigenvectorsTab, self).__init__(parent)
        self.main_window = parent
        self.user_prd_index = 1

        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        self.layoutL = QGridLayout()
        self.layoutL.setContentsMargins(20, 20, 20, 20)
        self.layoutL.setSpacing(10)

        self.layoutR = QGridLayout()
        self.layoutR.setContentsMargins(20, 20, 20, 20)
        self.layoutR.setSpacing(10)

        self.layoutB = QGridLayout()
        self.layoutB.setContentsMargins(20, 20, 20, 20)
        self.layoutB.setSpacing(10)

        self.widgetsL = QWidget()
        self.widgetsR = QWidget()
        self.widgetsB = QWidget()
        self.widgetsL.setLayout(self.layoutL)
        self.widgetsR.setLayout(self.layoutR)
        self.widgetsB.setLayout(self.layoutB)

        label_topos = QLabel("View Topos")
        label_topos.setMargin(0)
        label_topos.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.layoutR.addWidget(label_topos, 0, 8, 1, 4)


        def update_topos():  #refresh screen for new topos and anchors
            return
            # =================================================================
            # link topos directories and update with photos/videos:
            # =================================================================
            # create blank image if topos file doesn't exist
            # =================================================================
            picImg = Image.open(p.get_topos_path(self.user_prd_index, 1))
            picSize = picImg.size
            blank = np.zeros([picSize[0], picSize[1], 3], dtype=np.uint8)
            blank.fill(0)
            blank = QImage(blank, blank.shape[1], blank.shape[0], blank.shape[1] * 3, QImage.Format_RGB888)
            blankpix = QPixmap(blank)
            # =================================================================
            self.user_prd_index = self.entry_prd.value()

            topos_sum = 0
            for index in range(8):
                pic_path = p.get_topos_path(self.user_prd_index, index + 1)  # topos are 1 indexed
                if os.path.isfile(pic_path):
                    self.label_pic[index].setPixmap(QPixmap(pic_path))
                    self.button_pic[index].setDisabled(False)
                    topos_sum += 1
                else:
                    self.label_pic[index].setPixmap(QPixmap(blankpix))
                    self.button_pic[index].setDisabled(True)

            self.reactCoord1All[self.user_prd_index].setMaximum(topos_sum)
            self.reactCoord2All[self.user_prd_index].setMaximum(topos_sum)
            EigValCanvas().EigValRead()  #read in eigenvalue spectrum for current prd

            # =================================================================
            # Update trash section widgets:
            # =================================================================
            self.layoutL.removeWidget(self.trashAll[self.prd_hist])
            self.trashAll[self.prd_hist].close()
            self.layoutL.addWidget(self.trashAll[self.user_prd_index], 6, 5, 1, 2, QtCore.Qt.AlignCenter)
            self.trashAll[self.user_prd_index].show()

            # =================================================================
            # Update anchor section widgets:
            # =================================================================
            if p.user_dimensions == 1:
                self.layoutB.removeWidget(self.reactCoord1All[self.prd_hist])
                self.reactCoord1All[self.prd_hist].close()
                self.layoutB.addWidget(self.reactCoord1All[self.user_prd_index], 8, 2, 1, 1)
                self.reactCoord1All[self.user_prd_index].show()

                self.layoutB.removeWidget(self.senses1All[self.prd_hist])
                self.senses1All[self.prd_hist].close()
                self.layoutB.addWidget(self.senses1All[self.user_prd_index], 8, 3, 1, 1)
                self.senses1All[self.user_prd_index].show()

                self.layoutB.removeWidget(self.anchorsAll[self.prd_hist])
                self.anchorsAll[self.prd_hist].close()
                self.layoutB.addWidget(self.anchorsAll[self.user_prd_index], 8, 4, 1, 1)
                self.anchorsAll[self.user_prd_index].show()

            # add current prd to history list:
            self.prd_hist = self.user_prd_index


        self.viz2 = Mayavi_Rho(self)
        self.layoutL.addWidget(self.viz2.get_widget(), 0, 0, 6, 7)

        self.label_prd = QLabel('Projection Direction:')
        self.label_prd.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.layoutL.addWidget(self.label_prd, 6, 0, 1, 1)

        self.entry_prd = QSpinBox(self)
        self.entry_prd.setMinimum(1)
        self.entry_prd.setMaximum(1)
        self.entry_prd.setSuffix(f"  /  1")
        self.entry_prd.valueChanged.connect(self.update_prd)
        self.entry_prd.valueChanged.connect(update_topos)
        self.entry_prd.setToolTip('Change the projection direction of the current view above.')
        self.layoutL.addWidget(self.entry_prd, 6, 1, 1, 2)

        self.entry_pop = QDoubleSpinBox(self)
        self.entry_pop.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_pop.setToolTip('Total number of particles within current PD.')
        self.entry_pop.setDisabled(True)
        self.entry_pop.setDecimals(0)
        self.entry_pop.setMaximum(99999999)
        self.entry_pop.setSuffix(' images')
        self.layoutL.addWidget(self.entry_pop, 6, 3, 1, 2)

        self.label_pic = []
        self.button_pic = []
        subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        minSize = 1
        for i in range(8):
            label = QLabel()
            picpath = p.get_topos_path(self.user_prd_index, i + 1)
            label.setPixmap(QPixmap(picpath))
            label.setMinimumSize(minSize, minSize)
            label.setScaledContents(True)
            label.setAlignment(QtCore.Qt.AlignCenter)

            button = QPushButton('View %s%s' % ("\u03A8", str(i+1).translate(subscripts)), self)
            button.clicked.connect(lambda: self.CC_vid1(i+1))
            button.setToolTip('View 2d movie and related outputs.')

            self.label_pic.append(label)
            self.button_pic.append(button)

        for i in range(4):
            self.layoutR.addWidget(self.label_pic[i], 1, 8 + i, 1, 1)
            self.layoutR.addWidget(self.button_pic[i], 2, 8 + i, 1, 1)

        self.label_Hline = QLabel("")
        self.label_Hline.setMargin(0)
        self.label_Hline.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.layoutR.addWidget(self.label_Hline, 3, 8, 1, 4)

        for i in range(4, 8):
            self.layoutR.addWidget(self.label_pic[i], 4, 4 + i, 1, 1)
            self.layoutR.addWidget(self.button_pic[i], 5, 4 + i, 1, 1)

        button_bandwidth = QPushButton('Kernel Bandwidth')
        button_bandwidth.setDisabled(False)
        button_bandwidth.clicked.connect(self.bandwidth)
        self.layoutR.addWidget(button_bandwidth, 6, 8, 1, 1)

        button_eigSpec = QPushButton('Eigenvalue Spectrum')
        button_eigSpec.clicked.connect(lambda: eigSpectrum(self))
        self.layoutR.addWidget(button_eigSpec, 6, 9, 1, 1)

        button_viewAvg = QPushButton('2D Class Average')
        button_viewAvg.clicked.connect(lambda: classAvg(self))
        self.layoutR.addWidget(button_viewAvg, 6, 10, 1, 1)

        button_compareMov = QPushButton('Compare Movies')
        button_compareMov.clicked.connect(self.CC_vid2)
        self.layoutR.addWidget(button_compareMov, 6, 11, 1, 1)


        # confirm conformational coordinates:
        def anchorCheck():
            # first save PDs chosen for removal to file:
            self.trash_list = np.array([self.trashAll[i].isChecked() for i in range(1, P3.prd_total + 1)], dtype=bool)

            #### ZULU ####
            #{SECTION EVENTUALLY NEEDED FOR RE-DOING GRAPH IF TRASH LIST DIVIDED ANY ISLANDS INTO SUB-ISLANDS}

            #if len(self.trash_list) > 0:
            #trash_reviewed = T or F variable
            ##############

            trashDir = os.path.join(p.CC_dir, 'user_removals.txt')
            np.savetxt(trashDir, self.trash_list, fmt='%i', delimiter='\t')
            p.set_trash_list(self.trash_list)

            # save anchors to file:
            anch_sum = sum(map(lambda anchor: int(anchor.isChecked()), self.anchorsAll.values()))

            if anch_sum < anchorsMin:
                box = QMessageBox(self)
                box.setWindowTitle('%s Error' % progname)
                box.setText('<b>Input Error</b>')
                box.setIcon(QMessageBox.Information)
                box.setInformativeText('A minimum of %s PD anchors must be selected.' % anchorsMin)
                box.setStandardButtons(QMessageBox.Ok)
                box.setDefaultButton(QMessageBox.Ok)
                box.exec_()
                return

            prds = []
            CC1s = []
            S1s = []
            CC2s = []
            S2s = []
            colors = []
            self.anch_list = []

            idx = 0
            for i in range(1, P3.prd_total + 1):
                if self.anchorsAll[i].isChecked():
                    prds.append(int(i))
                    # CC1s:
                    CC1s.append(int(self.reactCoord1All[i].value()))
                    # S1s:
                    if self.senses1All[i].currentText() == 'S1: FWD':
                        S1s.append(int(1))
                    else:
                        S1s.append(int(-1))
                    # CC2s:
                    CC2s.append(int(self.reactCoord2All[i].value()))
                    # S2s:
                    if self.senses2All[i].currentText() == 'S2: FWD':
                        S2s.append(int(1))
                    else:
                        S2s.append(int(-1))
                    # colors:
                    colors.append(P3.col[int(i - 1)])
                    idx += 1

            if P3.user_dimensions == 1:
                self.anch_list = zip(prds, CC1s, S1s, colors)
            elif P3.user_dimensions == 2:
                self.anch_list = zip(prds, CC1s, S1s, CC2s, S2s, colors)

            keepers = np.array(P3.col)[~p.get_trash_list()]
            box = QMessageBox(self)
            # check if at least one anchor is selected for each color:
            if set(keepers) == set(colors):
                box.setWindowTitle('%s' % progname)
                box.setIcon(QMessageBox.Question)
                box.setText('<b>Confirm Conformational Coordinates</b>')

                msg = 'Performing this action will initiate Belief Propagation for the current \
                        PD anchors and generate the corresponding energy landscape and 3D volumes.\
                        <br /><br />\
                        Do you want to proceed?'
            else:
                box.setWindowTitle('%s Warning' % progname)
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')

                n_total, n_selected = len(set(keepers)), len(set(colors))
                msg = 'It is highly recommended that at least one anchor node is selected\
                        for each connected component (as seen via clusters of colored PDs on S2).\
                        <br /><br />\
                        Currently, only %s of %s connected components are satisfied in this manner,\
                        and thus, %s will be ignored during Belief Propagation.\
                        <br /><br />\
                        Do you want to proceed?' % (n_selected, n_total, n_total - n_selected)


            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setInformativeText(msg)

            if box.exec_() == QMessageBox.No:
                return

            self.anch_list, anch_zip = itertools.tee(self.anch_list)
            p.anch_list = list(anch_zip)  #prd,CC1,S1 for 1D; prd,CC1,S1,CC2,S2 for 2D
            anchInputs = os.path.join(p.CC_dir, 'user_anchors.txt')
            np.savetxt(anchInputs, p.anch_list, fmt='%i', delimiter='\t')
            gotoP5(self)
            if self.recompute == 0:
                self.btn_finOut.setText('Recompile Results')
                self.recompute = 1
                p.resProj = 6  # update progress
                p.save()  # send new GUI data to user parameters file
                return

            msg = 'Final outputs have already been computed for a previous\
                    selection of anchor nodes. To recompute final outputs\
                    with new anchor node settings, previous outputs must be\
                    overwritten.\
                    <br /><br />\
                    Do you want to proceed?'

            box = QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setText('<b>Overwrite Warning</b>')
            box.setIcon(QMessageBox.Warning)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setInformativeText(msg)
            if box.exec_() == QMessageBox.No:
                return

            P5.progress5.setValue(0)
            P5.progress6.setValue(0)
            Erg1dMain.progress7.setValue(0)
            P5.button_CC.setText('Find Conformational Coordinates')
            P5.button_erg.setText('Energy Landscape')
            Erg1dMain.button_traj.setText('Compute 3D Trajectories')
            Erg1dMain.reprepare = 0
            P5.entry_opt.setDisabled(False)
            P5.entry_proc.setDisabled(False)
            P5.entry_temp.setDisabled(False)
            P5.button_CC.setDisabled(False)
            P5.button_erg.setDisabled(True)
            Erg1dMain.button_traj.setDisabled(True)
            P5.button_toP6.setDisabled(True)
            tabs.setTabEnabled(5, False)

            p.resProj = 6  #revert progress back to before ever running FindConformationalCoords.py
            p.save()  #send new GUI data to user parameters file

            # =============================================
            # Hard-remove pre-existing folders:
            shutil.rmtree(p.CC_meas_dir)
            shutil.rmtree(p.CC_OF_dir)
            shutil.rmtree(p.EL_dir)
            prepOutDir = os.path.join(p.out_dir, 'bin')
            shutil.rmtree(prepOutDir)
            shutil.rmtree(p.traj_dir)

            os.makedirs(p.CC_meas_dir)
            os.makedirs(p.CC_OF_dir)
            os.makedirs(p.EL_dir)
            os.makedirs(p.OM_dir)
            os.makedirs(prepOutDir)
            os.makedirs(p.traj_dir)
            os.makedirs(p.CC_meas_prog)  #progress bar folder
            os.makedirs(p.EL_prog)  #progress bar folder
            # =============================================


        self.label_edgeAnchor = QLabel('')
        self.label_edgeAnchor.setMargin(5)
        self.label_edgeAnchor.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.layoutB.addWidget(self.label_edgeAnchor, 7, 0, 3, 7)

        self.label_anchor = QLabel('Set PD Anchors')
        self.label_anchor.setMargin(5)
        self.label_anchor.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.label_anchor.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.layoutB.addWidget(self.label_anchor, 7, 0, 1, 7)

        self.label_edgeCC = QLabel('')
        self.label_edgeCC.setMargin(5)
        self.label_edgeCC.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.layoutB.addWidget(self.label_edgeCC, 7, 8, 3, 4)

        self.label_reactCoord = QLabel('Confirm Conformational Coordinates')
        self.label_reactCoord.setMargin(5)
        self.label_reactCoord.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.label_reactCoord.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.layoutB.addWidget(self.label_reactCoord, 7, 8, 1, 4)

        self.btn_PDsele = QPushButton('   PD Selections   ', self)
        self.btn_PDsele.setToolTip('Review current PD selections.')
        self.btn_PDsele.clicked.connect(self.PDSeleViz)
        self.btn_PDsele.setDisabled(False)
        self.layoutB.addWidget(self.btn_PDsele, 8, 9, 1, 1, QtCore.Qt.AlignCenter)

        self.btn_finOut = QPushButton('   Compile Results   ', self)
        self.btn_finOut.setToolTip('Proceed to next section.')
        self.btn_finOut.clicked.connect(anchorCheck)
        self.btn_finOut.setDisabled(False)
        self.layoutB.addWidget(self.btn_finOut, 8, 10, 1, 1, QtCore.Qt.AlignLeft)

        # layout dividers:
        splitter1 = QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.widgetsL)
        splitter1.addWidget(self.widgetsR)
        splitter1.setStretchFactor(1, 1)

        splitter2 = QSplitter(QtCore.Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(self.widgetsB)

        self.layout.addWidget(splitter2)


    def PDSeleViz(self):
        global PDSele_window
        try:
            PDSele_window.close()
        except:
            pass
        PDSele_window = PDSeleMain()
        PDSele_window.setMinimumSize(10, 10)
        PDSele_window.setWindowTitle('Projection Direction Selections')
        PDSele_window.show()

    def on_button(self, n):
        print('Button {0} clicked'.format(n))

    def bandwidth(self):
        global BandwidthMain_window
        try:
            BandwidthMain_window.close()
        except:
            pass
        BandwidthMain_window = BandwidthMain()
        BandwidthMain_window.setMinimumSize(10, 10)
        BandwidthMain_window.setWindowTitle('Projection Direction %s' % (self.user_prd_index))
        BandwidthMain_window.show()

    def CC_vid1(self, n):
        self.gif_path = os.path.join(p.out_dir, 'topos', f'prd_{self.user_prd_index}', f'psi_{n}.gif')
        global prd_window
        try:
            prd_window.close()
        except:
            pass

        Manifold2dCanvas.coordsX = []
        Manifold2dCanvas.coordsY = []
        Manifold2dCanvas.eig_current = n
        eig_n_others = []
        eig_v_others = []
        index = 0
        for i in EigValCanvas.eig_v:  #find next highest eigenvalue
            index += 1
            if index != n:
                eig_n_others.append(EigValCanvas.eig_n[index - 1])
                eig_v_others.append(EigValCanvas.eig_v[index - 1])

        Manifold2dCanvas.eig_compare1 = eig_n_others[0]  #max eigenvalue (other than one selected)
        Manifold2dCanvas.eig_compare2 = eig_n_others[1]  #next highest eigenvalue from the above

        p.eig_current = Manifold2dCanvas.eig_current
        p.eig_compare1 = Manifold2dCanvas.eig_compare1

        VidCanvas.run = 0
        VidCanvas.img_paths = []
        VidCanvas.imgs = []
        VidCanvas.frames = 0
        prd_window = prd_Viz()
        prd_window.setWindowTitle('PD %s: Psi %s' % (self.user_prd_index, n))
        prd_window.show()

    def CC_vid2(self):
        global prd2_window
        try:
            prd2_window.close()
        except:
            pass

        Vid2Canvas.run = 0
        Vid2Canvas.gif_path1 = ''
        Vid2Canvas.gif_path2 = ''
        prd2_window = prd2_Viz()

        prd2_window.setWindowTitle('Compare NLSA Movies')
        prd2_window.show()

    # mayavi prd widget:
    def update_prd(self):
        # change angle of 3d plot to correspond with prd spinbox value:
        self.user_prd_index = self.entry_prd.value()
        prds = data_store.get_prds()
        phi = prds.phi_thresholded[self.user_prd_index - 1]
        theta = prds.theta_thresholded[self.user_prd_index - 1]
        self.viz2.update_view(azimuth=phi,
                              elevation=theta,
                              distance=self.viz2.view_angles())

        self.viz2.update_euler(phi, theta)
        population = prds.occupancy[(self.user_prd_index) - 1]
        self.entry_pop.setValue(population)

    def activate(self):
        prds = data_store.get_prds()
        self.entry_prd.setMaximum(prds.n_thresholded)
        self.entry_prd.setSuffix(f"  /  {prds.n_thresholded}")

        self.viz2.update_scene3(init=True)
        self.update_prd()
