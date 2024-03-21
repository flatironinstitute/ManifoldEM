import numpy as np

from traits.api import HasTraits, Instance, on_trait_change, Str, Range, Enum
from traitsui.api import View, Item, Group, HGroup, VGroup, TextEditor

from ManifoldEM.params import p
from ManifoldEM.data_store import data_store

import os
_disable_viz = bool(os.environ.get('MANIFOLD_DISABLE_VIZ', False))

if not _disable_viz:
    from mayavi import mlab
    from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
    from traitsui.api import View, Item, Group, HGroup, VGroup, TextEditor
else:
    class _Dummy:
        def __init__(self):
            pass
    mlab = None
    MlabSceneModel = _Dummy
    MayaviScene = None
    SceneEditor = None


class _Mayavi_Rho(HasTraits):
    """View of electrostatic potential map"""
    scene3 = Instance(MlabSceneModel, ())
    isosurface_level = Range(2, 9, 3, mode='enum')
    volume_alpha = Enum(1.0, .8, .6, .4, .2, 0.0)
    phi = Str
    theta = Str
    click_on = 0
    click_on_Eul = 0
    title = Str


    def _phi_default(self):
        return '%s%s' % (0, u"\u00b0")


    def _theta_default(self):
        return '%s%s' % (0, u"\u00b0")


    def _S2_scale_default(self):
        return float(1)


    def get_widget(self):
        return self.edit_traits(parent=self, kind='subpanel').control


    def view_angles(self):
        if _disable_viz:
            return
        zoom = self.scene3.mlab.view(figure=self.fig3)[2]
        return zoom


    def update_view(self, azimuth, elevation, distance):
        if _disable_viz:
            return
        self.scene3.mlab.view(azimuth=azimuth,
                              elevation=elevation,
                              distance=distance,
                              reset_roll=False,
                              figure=self.fig3)


    def update_euler_view(self, phi, theta):
        self.phi = '%s%s' % (round(phi, 2), u"\u00b0")
        self.theta = '%s%s' % (round(theta, 2), u"\u00b0")


    def __init__(self, parent):
        super(_Mayavi_Rho, self).__init__()
        self.parent = parent


    @on_trait_change('volume_alpha,isosurface_level')
    def update_scene3(self, init=False):
        if _disable_viz:
            return
        # store current camera info:
        view = self.scene3.mlab.view()
        roll = self.scene3.mlab.roll()
        self.fig3 = mlab.figure(3)
        self.scene3.background = (0.0, 0.0, 0.0)

        prds = data_store.get_prds()
        s2_positions = prds.pos_thresholded

        if init:
            mlab.clf(figure=self.fig3)

            # =================================================================
            # Volume (contour):
            # =================================================================
            import mrcfile
            with mrcfile.open(p.avg_vol_file, mode='r') as mrc:
                df_vol = mrc.data

            mirror = df_vol[..., ::-1]

            cplot = mlab.contour3d(mirror, contours=self.isosurface_level, color=(0.5, 0.5, 0.5), figure=self.fig3)
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
                                  figure=self.fig3)
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
                                  figure=self.fig3)

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
            trash_pos = s2_positions[:, list(prds.trash_ids)]
            tplot = mlab.points3d(trash_pos[0, :],
                                  trash_pos[1, :],
                                  trash_pos[2, :],
                                  scale_mode='none',
                                  scale_factor=0.06,
                                  figure=self.fig3)

            tplot.actor.property.backface_culling = True
            tplot.glyph.color_mode = 'no_coloring'
            tplot.actor.property.color = (0.0, 0.0, 0.0)
            tplot.actor.actor.scale = scale_vec
            tplot.actor.actor.origin = np.zeros(3)
            tplot.actor.actor.position = np.zeros(3)

            self.trash_update = tplot.mlab_source

        else:  #only update anchors
            anchor_pos = s2_positions[:, list(prds.anchors.keys())]
            trash_pos = s2_positions[:, list(prds.trash_ids)]
            self.anchors_update.reset(x=anchor_pos[0, :], y=anchor_pos[1, :], z=anchor_pos[2, :])
            self.trash_update.reset(x=trash_pos[0, :], y=trash_pos[1, :], z=trash_pos[2, :])

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
                angles = self.scene3.mlab.view(figure=self.fig3)
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

                # update view:
                phi, theta = prds.phi_thresholded[idx], prds.theta_thresholded[idx]
                self.update_euler_view(phi, theta)
                self.parent.entry_prd.setValue(idx + 1)  #update prd and thus topos


        self.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        self.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)

        # live update of Euler angles:
        def press_callback_Eul(vtk_obj, event):  #left mouse down callback
            self.click_on_Eul = 1

        def hold_callback_Eul(vtk_obj, event):  #camera rotate callback
            if self.click_on_Eul > 0:
                viewS2 = self.scene3.mlab.view(figure=self.fig3)
                self.phi = '%s%s' % (round(viewS2[0], 2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1], 2), u"\u00b0")

        def release_callback_Eul(vtk_obj, event):  #left mouse release callback
            if self.click_on_Eul == 1:
                self.click_on_Eul = 0

        self.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback_Eul)
        self.fig3.scene.scene.interactor.add_observer('InteractionEvent', hold_callback_Eul)
        self.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback_Eul)


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
                     editor=SceneEditor(scene_class=MayaviScene) if not _disable_viz else None,
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
                    Item('isosurface_level',
                         springy=True,
                         show_label=True,
                         tooltip='Change the isosurface level of the volume map above.'),
                ),
                show_border=False,
                orientation='vertical'),
        ),
        resizable=True,
    )
