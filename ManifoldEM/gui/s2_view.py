import mrcfile
import pickle

import numpy as np
from scipy import stats

import subprocess
has_gl = not bool(subprocess.run('glxinfo', capture_output=True).returncode)
if has_gl:
    print("GL context found, using Mayavi frontend")
    from mayavi import mlab
    from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
    from traitsui.api import View, Item, Group, HGroup, VGroup, TextEditor
else:
    print("No GL context available, falling back on matplotlib for visualization")
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass
    mlab = _Dummy
    MlabSceneModel = _Dummy
    MayaviScene = _Dummy
    SceneEditor = _Dummy
    View = _Dummy
    Item = _Dummy
    Group = _Dummy
    HGroup = _Dummy
    VGroup = _Dummy
    TextEditor = _Dummy

from traits.api import Instance, HasTraits, List, Enum, Button, Str, Range, Int, observe

from ManifoldEM.params import p
from .threshold_view import ThresholdView
from .utils import threshold_pds

from PyQt5.QtWidgets import QWidget

def press_callback(parent, vtk_obj, event):  # left mouse down callback
    parent.click_on = 1


def hold_callback(parent, scene, fig, vtk_obj, event):  # camera rotate callback
    if parent.click_on > 0:
        viewS2 = scene.mlab.view(figure=fig)
        parent.phi = f"{round(viewS2[0], 2)}\u00b0"
        parent.theta = f"{round(viewS2[1], 2)}\u00b0"


def release_callback(parent, vtk_obj, event):  # left mouse release callback
    if parent.click_on == 1:
        parent.click_on = 0

class S2ViewBase:
    def load_data(self):
        self.update_S2_params()
        with open(p.tess_file, 'rb') as f:
            data = pickle.load(f)
            self.S2_data = data['S2']

        self.S2_density_all = threshold_pds()
        self.get_volume_data()

    def get_volume_data(self):
        if self.df_vol is None:
            with mrcfile.open(p.avg_vol_file, mode='r') as mrc:
                self.df_vol = mrc.data

    def update_S2_params(self):
        self.isosurface_level = int(p.visualization_params['S2_isosurface_level'])
        self.S2_scale = float(p.visualization_params['S2_scale'])
        self.S2_density = int(p.visualization_params['S2_density'])


class S2ViewMayavi(HasTraits, S2ViewBase):
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

    def get_widget(self):
        return self.edit_traits(parent=self, kind='subpanel').control

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

        splot.actor.actor.scale = self.S2_scale * len(self.df_vol) / np.sqrt(2.0) * np.ones(3)
        splot.actor.property.backface_culling = True
        splot.mlab_source.reset

        splot.module_manager.scalar_lut_manager.scalar_bar_widget.repositionable = False
        splot.module_manager.scalar_lut_manager.scalar_bar_widget.resizable = False

        # reposition camera to previous:
        self.scene1.mlab.view(*view)
        self.scene1.mlab.roll(roll)

        self.fig1.scene.scene.interactor.add_observer('LeftButtonPressEvent',
                                                      lambda x, y: press_callback(self, x, y))
        self.fig1.scene.scene.interactor.add_observer('InteractionEvent',
                                                      lambda x, y: hold_callback(self, self.scene1, self.fig1, x, y))
        self.fig1.scene.scene.interactor.add_observer('EndInteractionEvent',
                                                      lambda x, y: release_callback(self, x, y))

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

        # reposition camera to previous:
        mlab.view(view[0], view[1], len(self.df_vol) * 2, view[3])  #zoom out based on MRC volume dimensions
        mlab.roll(roll)

        self.fig2.scene.scene.interactor.add_observer('LeftButtonPressEvent',
                                                      lambda x, y: press_callback(self, x, y))
        self.fig2.scene.scene.interactor.add_observer('InteractionEvent',
                                                      lambda x, y: hold_callback(self, self.scene2, self.fig2, x, y))
        self.fig2.scene.scene.interactor.add_observer('EndInteractionEvent',
                                                      lambda x, y: release_callback(self, x, y))

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

class S2ViewMatplotlib(QWidget, S2ViewBase):
    def __init__(self, parent=None):
        super(S2ViewMatplotlib, self).__init__(parent)
        self.df_vol = None

    def get_widget(self):
        return self

    def update_scene1(self, event):
        pass

    def update_scene2(self, event):
        pass

if has_gl:
    S2View = S2ViewMayavi
else:
    S2View = S2ViewMatplotlib
