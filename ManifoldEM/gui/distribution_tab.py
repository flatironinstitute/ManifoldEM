from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QMessageBox,
                             QInputDialog, QDoubleSpinBox, QGridLayout, QWidget, QMainWindow)

from numbers import Number
from typing import Tuple, Union

import pickle

import numpy as np
from ManifoldEM import p

from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from scipy import stats

import pandas
import mrcfile

from traits.api import Instance, HasTraits, List, Enum, Button, Str, Range, Int, observe
from traitsui.api import View, Item, Group, HGroup, VGroup, TextEditor

class ThresholdView(QMainWindow):
    def __init__(self):
        super(ThresholdView, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&PD Thresholding', self.guide_threshold)

    def initUI(self):
        pass

def threshold_pds():
    print('\tlow threshold: %s' % p.PDsizeThL)
    print('\thigh threshold: %s' % p.PDsizeThH)
    print('\tthresholding PDs...')
    with open(p.tess_file, 'rb') as f:
        data = pickle.load(f)

    # all tessellated bins:
    totalPrDs = int(np.shape(data['CG1'])[0])
    mid = data['CG1'].shape[0] // 2
    NC1 = data['NC'][:int(mid)]
    NC2 = data['NC'][int(mid):]

    all_PrDs = []
    all_occ = []
    thresh_PrDs = []
    thresh_occ = []
    if len(NC1) >= len(NC2):  #first half of S2
        pd_all = 1
        pd = 1
        for i in NC1:
            all_PrDs.append(pd_all)
            all_occ.append(i)
            if i >= p.PDsizeThL:
                thresh_PrDs.append(pd)
                if i > p.PDsizeThH:
                    thresh_occ.append(p.PDsizeThH)
                else:
                    thresh_occ.append(i)
                pd += 1
            pd_all += 1
    else:  #second half of S2
        pd_all = 1
        pd = 1
        for i in NC2:
            all_PrDs.append(pd_all)
            all_occ.append(i)
            if i >= p.PDsizeThL:
                thresh_PrDs.append(pd)
                if i > p.PDsizeThH:
                    thresh_occ.append(p.PDsizeThH)
                else:
                    thresh_occ.append(i)
                pd += 1
            pd_all += 1

    # read points from tesselated sphere:
    PrD_map1_eul = pandas.read_csv(p.ref_ang_file1, delimiter='\t')

    all_phi = PrD_map1_eul['phi']
    all_theta = PrD_map1_eul['theta']

    # ad hoc ratios to make sure S2 volume doesn't freeze-out due to too many particles to plot:
    ratio1 = float(sum(all_occ)) / 2000
    ratio2 = float(sum(all_occ)) / 5000
    S2_density_all = [5, 10, 25, 50, 100, 250, 500, 1000, 10000, 100000]
    S2_density_all = list(filter(lambda a: a < int(sum(all_occ)), S2_density_all))
    S2_density_all = list(filter(lambda a: a > int(ratio2), S2_density_all))

    return [int(el) for el in S2_density_all]


class S2View(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    S2_scale_all = List([.2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    S2_scale = Enum(1.0, values='S2_scale_all')
    display_angle = Button('Display Euler Angles')
    phi = Str("0\u00b0")
    theta = Str("0\u00b0")
    display_thresh = Button('PD Thresholding')
    isosurface_level = Range(2, 9, 3, mode='enum')
    S2_density_all = List([5, 10, 25, 50, 100, 250, 500, 1000, 10000, 100000])
    S2_density = Enum(1000, values='S2_density_all')

    click_on = Int(0)
    titleLeft = Str('S2 Orientation Distribution')
    titleRight = Str('Electrostatic Potential Map')
    title = Str('Electrostatic Potential Map')

    def __init__(self):
        HasTraits.__init__(self)
        self.df_vol = None

    def load_data(self):
        self.update_S2_params()
        with open(p.tess_file, 'rb') as f:
            data = pickle.load(f)
            self.S2_data = data['S2']

        self.S2_density_all = threshold_pds()
        self.get_volume_data()

    def get_volume_data(self):
        if self.df_vol is None:
            with mrcfile.open(p.avg_vol_file, mode='r+') as mrc:
                mrc.header.mapc = 1
                mrc.header.mapr = 2
                mrc.header.maps = 3
                self.df_vol = mrc.data

    def update_S2_params(self):
        self.isosurface_level = int(p.visualization_params['S2_isosurface_level'])
        self.S2_scale = float(p.visualization_params['S2_scale'])
        self.S2_density = int(p.visualization_params['S2_density'])

    def sync_params(self):
        p.visualization_params['S2_scale'] = self.S2_scale
        p.visualization_params['S2_density'] = self.S2_density
        p.visualization_params['S2_isosurface_level'] = self.isosurface_level
        p.save()

    @observe('display_angle')
    def view_anglesP2(self, event):
        viewS2 = self.scene1.mlab.view(figure=self.fig1)
        azimuth = viewS2[0]  # phi: 0-360
        elevation = viewS2[1]  # theta: 0-180
        print_anglesP2(azimuth, elevation)

    @observe('S2_scale, S2_density')  #S2 Orientation Sphere
    def update_scene1(self, event):
        # store current camera info:
        view = self.scene1.mlab.view()
        roll = self.scene1.mlab.roll()

        self.fig1 = mlab.figure(1, bgcolor=(.5, .5, .5))
        self.fig2 = mlab.figure(2, bgcolor=(.5, .5, .5))

        mlab.clf(figure=self.fig1)

        x1 = self.S2_data[0, ::self.S2_density]
        y1 = self.S2_data[1, ::self.S2_density]
        z1 = self.S2_data[2, ::self.S2_density]
        values = np.array([x1, y1, z1])
        try:
            kde = stats.gaussian_kde(values)
            d1 = kde(values)  # density
            d1 /= d1.max()  # relative density, max=1

            splot = mlab.points3d(x1, y1, z1, d1,
                                  scale_mode='none',
                                  scale_factor=0.05,
                                  figure=self.fig1)
            cbar = mlab.scalarbar(title='Relative\nDensity\n', orientation='vertical',
                                  nb_labels=3, label_fmt='%.1f')
        except:
            splot = mlab.points3d(x1, y1, z1, scale_mode='none', scale_factor=0.05, figure=self.fig1)

        #####################
        # align-to-grid data:
        phi, theta = np.mgrid[0:np.pi:11j, 0:2 * np.pi:11j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
        testPlot.actor.actor.scale = np.array([50, 50, 50])
        testPlot.actor.property.opacity = 0
        #####################

        splot.actor.actor.scale = self.S2_scale * len(self.df_vol) / np.sqrt(2.0) * np.ones(3)
        splot.actor.property.backface_culling = True
        splot.mlab_source.reset

        splot.module_manager.scalar_lut_manager.scalar_bar_widget.repositionable = False
        splot.module_manager.scalar_lut_manager.scalar_bar_widget.resizable = False

        # reposition camera to previous:
        self.scene1.mlab.view(*view)
        self.scene1.mlab.roll(roll)

        def press_callback(vtk_obj, event):  # left mouse down callback
            self.click_on = 1

        def hold_callback(vtk_obj, event):  # camera rotate callback
            if self.click_on > 0:
                viewS2 = self.scene1.mlab.view(figure=self.fig1)
                self.phi = '%s%s' % (round(viewS2[0], 2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1], 2), u"\u00b0")

        def release_callback(vtk_obj, event):  # left mouse release callback
            if self.click_on == 1:
                self.click_on = 0

        self.fig1.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        self.fig1.scene.scene.interactor.add_observer('InteractionEvent', hold_callback)
        self.fig1.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)

        self.sync_params()

    @observe('isosurface_level')  #Electrostatic Potential Map
    def update_scene2(self, event):
        # store current camera info:
        view = mlab.view()
        roll = mlab.roll()

        self.fig1 = mlab.figure(1, bgcolor=(.5, .5, .5))
        self.fig2 = mlab.figure(2, bgcolor=(.5, .5, .5))

        mlab.sync_camera(self.fig1, self.fig2)
        mlab.sync_camera(self.fig2, self.fig1)

        mlab.clf(figure=self.fig2)

        if p.relion_data:
            mirror = self.df_vol[..., ::-1]
            cplot = mlab.contour3d(mirror,
                                   contours=self.isosurface_level,
                                   color=(0.9, 0.9, 0.9),
                                   figure=self.fig2)
            cplot.actor.actor.orientation = np.array([0., -90., 0.])

        else:
            cplot = mlab.contour3d(self.df_vol,
                                   contours=self.isosurface_level,
                                   color=(0.9, 0.9, 0.9),
                                   figure=self.fig2)

        cplot.actor.actor.origin = np.array([len(self.df_vol) / 2, len(self.df_vol) / 2, len(self.df_vol) / 2])
        cplot.actor.actor.position = np.array([-len(self.df_vol) / 2, -len(self.df_vol) / 2, -len(self.df_vol) / 2])

        cplot.actor.property.backface_culling = True
        cplot.compute_normals = False
        cplot.mlab_source.reset

        #####################
        # align-to-grid data:
        phi, theta = np.mgrid[0:np.pi:11j, 0:2 * np.pi:11j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
        testPlot.actor.actor.scale = np.array([50, 50, 50])
        testPlot.actor.property.opacity = 0
        ####################

        # reposition camera to previous:
        mlab.view(view[0], view[1], len(self.df_vol) * 2, view[3])  #zoom out based on MRC volume dimensions
        mlab.roll(roll)

        def press_callback(vtk_obj, event):  # left mouse down callback
            self.click_on = 1

        def hold_callback(vtk_obj, event):  # camera rotate callback
            if self.click_on > 0:
                viewS2 = self.scene2.mlab.view(figure=self.fig2)
                self.phi = '%s%s' % (round(viewS2[0], 2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1], 2), u"\u00b0")

        def release_callback(vtk_obj, event):  # left mouse release callback
            if self.click_on == 1:
                self.click_on = 0

        self.fig2.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        self.fig2.scene.scene.interactor.add_observer('InteractionEvent', hold_callback)
        self.fig2.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)

        self.sync_params()

    @observe('display_thresh')
    def GCsViewer(self, event):
        global GCs_window
        try:
            GCs_window.close()
        except:
            pass
        GCs_window = ThresholdView()
        GCs_window.setMinimumSize(10, 10)
        GCs_window.setWindowTitle('Projection Direction Thresholding')
        GCs_window.show()

    view = View(
        VGroup(
            HGroup(  # HSplit
                Group(
                    Item('titleLeft',
                         springy=False,
                         show_label=False,
                         style='readonly',
                         style_sheet='*{font-size:12px; qproperty-alignment:AlignCenter}'),
                    Item(
                        'scene1',
                        editor=SceneEditor(scene_class=MayaviScene),
                        height=1,
                        width=1,
                        show_label=False,
                        springy=True,
                    ),
                ),
                Group(
                    Item('titleRight',
                         springy=False,
                         show_label=False,
                         style='readonly',
                         style_sheet='*{font-size:12px; qproperty-alignment:AlignCenter}'),
                    Item(
                        'scene2',
                        editor=SceneEditor(scene_class=MayaviScene),
                        height=1,
                        width=1,
                        show_label=False,
                        springy=True,
                    ),
                ),
            ),
            HGroup(
                HGroup(Item('display_thresh',
                            springy=True,
                            show_label=False,
                            tooltip='Display the occupancy of each PD.'),
                       Item('S2_scale',
                            springy=True,
                            show_label=True,
                            tooltip='Change the relative scale of S2 with respect to the volume map above.'),
                       Item('S2_density',
                            springy=True,
                            show_label=True,
                            tooltip='Density of available points displayed on S2.'),
                       show_border=True,
                       orientation='horizontal'),
                HGroup(
                    Item(
                        'phi',
                        springy=True,
                        show_label=True,
                        editor=TextEditor(evaluate=float),
                        enabled_when='phi == float(0)',
                    ),
                    Item(
                        'theta',
                        springy=True,
                        show_label=True,
                        editor=TextEditor(evaluate=float),
                        enabled_when='phi == float(0)',
                    ),
                    Item('isosurface_level',
                         springy=True,
                         show_label=True,
                         tooltip='Change the isosurface level of the volume map above.'),
                    show_border=True,
                    orientation='horizontal'),
            ),
        ),
        resizable=True,
    )


class DistributionTab(QWidget):
    def __init__(self, parent=None):
        super(DistributionTab, self).__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.viz = S2View()
        self.ui_element = self.viz.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui_element, 0, 0, 1, 6)

        # next page:
        self.label_Hline = QLabel("")  #aesthetic line left
        self.label_Hline.setMargin(20)
        self.label_Hline.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        layout.addWidget(self.label_Hline, 2, 0, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()

        self.label_Hline = QLabel("")  #aesthetic line right
        self.label_Hline.setMargin(20)
        self.label_Hline.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        layout.addWidget(self.label_Hline, 2, 4, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()

        # if Graphics is False:
        #     self.button_threshPD = QPushButton('PD Thresholding')
        #     layout.addWidget(self.button_threshPD, 1, 2, 1, 2)
        #     self.button_threshPD.show()  #FELIX

        self.button_binPart = QPushButton('Bin Particles')
        self.button_binPart.setToolTip('Proceed to embedding.')
        layout.addWidget(self.button_binPart, 2, 2, 1, 2)
        self.button_binPart.show()
        self.show()

    def activate(self):
        self.viz.load_data()
        self.viz.update_scene1(None)
        self.viz.update_scene2(None)
