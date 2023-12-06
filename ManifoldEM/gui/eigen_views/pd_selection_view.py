import os
import numpy as np

import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QDialog, QLabel, QFrame, QPushButton, QTabWidget,
                             QLayout, QGridLayout, QDesktopWidget, QTableWidget, QTableWidgetItem)

from ManifoldEM.data_store import data_store
from ManifoldEM.params import p

# FIXME Loads/saves for prds/trash overwrite other progress
class _PDSelectorWindow(QMainWindow):
    def __init__(self, parent=None, eigenvector_view=None):
        super(_PDSelectorWindow, self).__init__(parent)
        self.eigenvector_view = eigenvector_view
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&PD Selections', self.guide_PDSele)


    def guide_PDSele(self):
        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Help')
        box.setText('<b>PD Selections</b>')
        box.setInformativeText("<span style='font-weight:normal;'>\
                                On the <i>PD Editor</i> tab, all PD selections can be reviewed in list form, reset, saved, or loaded via the\
                                corresponding buttons.\
                                <br /><br />\
                                On the <i>PD Viewer</i> tab, the polar coordinates plot shows the location of each user-defined PD classification: \
                                <i>To Be Determined</i> (TBD), <i>Anchors</i>, and <i>Removals</i>.\
                                </span>")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()


    def initUI(self):
        self.sele_tab1 = PDEditorCanvas(self, eigenvector_view=self.eigenvector_view)
        self.sele_tab2 = PDViewerCanvas(self)
        self.sele_tabs = QTabWidget(self)
        self.sele_tabs.addTab(self.sele_tab1, 'PD Editor')
        self.sele_tabs.addTab(self.sele_tab2, 'PD Viewer')

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(self.sele_tabs)
        self.show()

        self.sele_tabs.currentChanged.connect(self.onTabChange)  #signal for tab changed via direct click


    def onTabChange(self, i):
        if i == 1:  #signals when view switched to tab 2
            prds = data_store.get_prds()
            trash_ids = sorted(list(prds.trash_ids))
            anchor_ids = sorted(list(prds.anchor_ids))
            unlabeled_ids = sorted(list(set(range(prds.n_thresholded)) - set(anchor_ids) - set(trash_ids)))
            phi_all0 = (prds.phi_thresholded - 180.0) * np.pi / 180.0
            theta_all0 = prds.theta_thresholded
            cluster_all0 = prds.cluster_ids

            phi_anch = phi_all0[anchor_ids]
            theta_anch = theta_all0[anchor_ids]

            phi_trash = phi_all0[trash_ids]
            theta_trash = theta_all0[trash_ids]

            phi_all = phi_all0[unlabeled_ids]
            theta_all = theta_all0[unlabeled_ids]
            cluster_all = cluster_all0[unlabeled_ids]

            def format_coord(x, y):  #calibrates toolbar coordinates
                return 'Phi={:1.2f}, Theta={:1.2f}'.format(((x * 180) / np.pi) - 180, y)

            # replot PDSeleCanvas:
            self.sele_tab2.axes.clear()
            self.sele_tab2.axes.format_coord = format_coord

            theta_labels = ['±180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']
            self.sele_tab2.axes.set_ylim(0, 180)
            self.sele_tab2.axes.set_yticks(np.arange(0, 180, 20))
            self.sele_tab2.axes.set_xticklabels(theta_labels)
            for tick in self.sele_tab2.axes.xaxis.get_major_ticks():
                tick.label1.set_fontsize(4)
            for tick in self.sele_tab2.axes.yaxis.get_major_ticks():
                tick.label1.set_fontsize(4)
            self.sele_tab2.axes.grid(alpha=.2)
            self.sele_tab2.axes.tick_params(pad=.3)  #distance of theta ticks from circle's edge

            pd_select_all = self.sele_tab2.axes.scatter(phi_all, theta_all, edgecolor='k',\
                                                        linewidth=.1, c=cluster_all, s=5, label='TBD')
            pd_select_all.set_alpha(0.75)
            pd_select_anch = self.sele_tab2.axes.scatter(phi_anch, theta_anch, edgecolor='k',\
                                                         linewidth=.3, c='lightgray', s=5, marker='D', label='Anchor')
            pd_select_anch.set_alpha(0.75)
            pd_select_trash = self.sele_tab2.axes.scatter(phi_trash, theta_trash, edgecolor='k',\
                                                          linewidth=.5, c='k', s=5, marker='x', label='Removal') #x or X
            pd_select_trash.set_alpha(1.)

            self.sele_tab2.axes.legend(loc='best', prop={'size': 4})

            self.sele_tab2.canvas.draw()


class PDEditorCanvas(QDialog):
    def __init__(self, parent=None, eigenvector_view=None):
        super(PDEditorCanvas, self).__init__(parent)
        self.eigenvector_view = eigenvector_view

        label_edgeLarge1 = QLabel('')
        label_edgeLarge1.setMargin(20)
        label_edgeLarge1.setLineWidth(1)
        label_edgeLarge1.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        label_anch = QLabel('PD Anchors:')
        label_anch.setMargin(20)
        label_anch.setFrameStyle(QFrame.Box | QFrame.Sunken)
        label_anch.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge2 = QLabel('')
        label_edgeLarge2.setMargin(20)
        label_edgeLarge2.setLineWidth(1)
        label_edgeLarge2.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        label_trash = QLabel('PD Removals:')
        label_trash.setMargin(20)
        label_trash.setFrameStyle(QFrame.Box | QFrame.Sunken)
        label_trash.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge3 = QLabel('')
        label_edgeLarge3.setMargin(20)
        label_edgeLarge3.setLineWidth(1)
        label_edgeLarge3.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        label_occ = QLabel('PD Occupancies:')
        label_occ.setMargin(20)
        label_occ.setFrameStyle(QFrame.Box | QFrame.Sunken)
        label_occ.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge4 = QLabel('')
        label_edgeLarge4.setMargin(20)
        label_edgeLarge4.setLineWidth(1)
        label_edgeLarge4.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        label_rebed = QLabel('PD Embeddings:')
        label_rebed.setMargin(20)
        label_rebed.setFrameStyle(QFrame.Box | QFrame.Sunken)
        label_rebed.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.btn_anchList = QPushButton('List Anchors')
        self.btn_anchList.clicked.connect(self.view_anchor_table)
        self.btn_anchReset = QPushButton('Reset Anchors')
        self.btn_anchReset.clicked.connect(self.anchor_reset)
        self.btn_anchSave = QPushButton('Save Anchors')
        self.btn_anchSave.clicked.connect(self.anchor_save)
        self.btn_anchLoad = QPushButton('Load Anchors')
        self.btn_anchLoad.clicked.connect(self.anchor_load)

        self.btn_trashList = QPushButton('List Removals')
        self.btn_trashList.clicked.connect(self.view_trash_table)
        self.btn_trashReset = QPushButton('Reset Removals')
        self.btn_trashReset.clicked.connect(self.trash_reset)
        self.btn_trashSave = QPushButton('Save Removals')
        self.btn_trashSave.clicked.connect(self.trash_save)
        self.btn_trashLoad = QPushButton('Load Removals')
        self.btn_trashLoad.clicked.connect(self.trash_load)

        self.btn_occList = QPushButton('List Occupancies')
        self.btn_occList.clicked.connect(self.view_occupancy_table)
        self.btn_rebedList = QPushButton('List Re-embeddings')
        self.btn_rebedList.clicked.connect(self.view_reembeddings_table)

        # forced space bottom:
        label_spaceBtm = QLabel("")
        label_spaceBtm.setMargin(0)
        label_spaceBtm.show()

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        layout.addWidget(label_edgeLarge1, 0, 0, 1, 6, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_anch, 0, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchList, 0, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchReset, 0, 2, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchSave, 0, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchLoad, 0, 4, 1, 1, QtCore.Qt.AlignVCenter)

        layout.addWidget(label_edgeLarge2, 2, 0, 1, 6, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_trash, 2, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashList, 2, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashReset, 2, 2, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashSave, 2, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashLoad, 2, 4, 1, 1, QtCore.Qt.AlignVCenter)

        layout.addWidget(label_edgeLarge3, 4, 0, 1, 3, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_occ, 4, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_occList, 4, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_edgeLarge4, 4, 3, 1, 3, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_rebed, 4, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_rebedList, 4, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_spaceBtm, 5, 1, 3, 6, QtCore.Qt.AlignVCenter)

        self.setLayout(layout)


    def view_anchor_table(self):
        prds = data_store.get_prds()
        idx_sum = len(prds.anchor_ids)

        if idx_sum == 0:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Error')
            box.setText('<b>Input Error</b>')
            box.setIcon(QMessageBox.Information)
            box.setInformativeText('No PD anchors have been selected.\
                                    <br /><br />\
                                    Select anchors using the <i>Set PD Anchors</i> box\
                                    on the left side of the <i>Eigenvectors</i> tab.')
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()
            return

        prds = data_store.get_prds()
        if p.dim == 1:
            headers = ['PD', 'CC', 'Sense', 'Color']
            values = [(id + 1, anchor.CC, anchor.sense.value, prds.cluster_ids[id]) for (id, anchor) in prds.anchors.items()]
        else:
            raise ValueError("Invalid dimension")

        self.anchor_table = TableView(headers, values, title='Review PD Anchors')

        sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
        self.anchor_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
        self.anchor_table.show()


    # reset assignments of all PD anchors:
    def anchor_reset(self):
        box = QMessageBox(self)
        self.setWindowTitle('Reset PD Anchors')
        box.setText('<b>Reset Warning</b>')
        box.setIcon(QMessageBox.Warning)
        msg = "Performing this action will deselect all active anchors. "\
              "The user-defined values set within each (if any) will be lost.\n"\
              "Would you like to proceed?"

        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setInformativeText(msg)
        reply = box.exec_()

        if reply == QMessageBox.Yes:
            prds = data_store.get_prds()
            prds.anchors.clear()

            if self.eigenvector_view is not None:
                self.eigenvector_view.on_prd_change()


    def anchor_save(self):
        data_store.get_prds().save()

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Save Current Anchors')
        box.setIcon(QMessageBox.Information)
        box.setText('<b>Saving Complete</b>')
        msg = 'Current anchor selections have been saved.'
        box.setStandardButtons(QMessageBox.Ok)
        box.setInformativeText(msg)
        box.exec_()


    def anchor_load(self):
        data_store.get_prds().load()

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Load Previous Anchors')
        box.setIcon(QMessageBox.Information)
        box.setText('<b>Loading Complete</b>')
        msg = 'Previous anchor selections have been loaded on the <i>Eigenvectors</i> tab.'
        box.setStandardButtons(QMessageBox.Ok)
        box.setInformativeText(msg)

        if self.eigenvector_view is not None:
            self.eigenvector_view.on_prd_change()


    def view_trash_table(self):
        prds = data_store.get_prds()

        if not len(prds.trash_ids):
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Error')
            box.setText('<b>Input Error</b>')
            box.setIcon(QMessageBox.Information)
            box.setInformativeText('No PDs have been selected for removal.\
                                    <br /><br />\
                                    Select PDs for removal using the <i>Remove PD</i> option\
                                    on the left side of the <i>Eigenvectors</i> tab.')
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()
            return

        headers = ['PD']
        values = sorted([(a + 1,) for a in prds.trash_ids])
        self.trash_table = TableView(headers, values, title='PD Removals')
        sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
        self.trash_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
        self.trash_table.show()


    def trash_reset(self):
        box = QMessageBox(self)
        self.setWindowTitle('Reset PD Removals')
        box.setText('<b>Reset Warning</b>')
        box.setIcon(QMessageBox.Warning)
        msg = 'Performing this action will deselect all active removals.\
                <br /><br />\
                Do you want to proceed?'

        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setInformativeText(msg)

        if box.exec_() == QMessageBox.No:
            return

        data_store.get_prds().trash_ids.clear()

        if self.eigenvector_view is not None:
            self.eigenvector_view.on_prd_change()


    def trash_save(self):
        data_store.get_prds().save()

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Save Current Removed PDs')
        box.setIcon(QMessageBox.Information)
        box.setText('<b>Saving Complete</b>')
        msg = 'Current PD removals have been saved.'
        box.setStandardButtons(QMessageBox.Ok)
        box.setInformativeText(msg)
        box.exec_()


    def trash_load(self):
        data_store.get_prds().load()

        box = QMessageBox(self)
        box.setWindowTitle('ManifoldEM Load previous trash')
        box.setIcon(QMessageBox.Information)
        box.setText('<b>Loading Complete</b>')
        msg = 'Previous removal selections have been loaded on the <i>Eigenvectors</i> tab.'
        box.setStandardButtons(QMessageBox.Ok)
        box.setInformativeText(msg)
        box.exec_()

        if self.eigenvector_view is not None:
            self.eigenvector_view.on_prd_change()


    def view_occupancy_table(self):
        prds = data_store.get_prds()

        headers = ['PD index', 'Occupancy', 'Cluster']
        values = sorted(zip(range(1, prds.n_thresholded + 1), prds.occupancy, prds.cluster_ids), key=lambda x: x[1], reverse=True)

        self.occupancy_table = TableView(headers, values)
        sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
        self.occupancy_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
        self.occupancy_table.show()


    def view_reembeddings_table(self):
        print("FIXME: re-embeddings view not implemented")


class PDViewerCanvas(QDialog):
    def __init__(self, parent=None):
        super(PDViewerCanvas, self).__init__(parent)
        self.left = 10
        self.top = 10

        # create canvas and plot data:
        PDViewerCanvas.figure = Figure(dpi=200)
        PDViewerCanvas.canvas = FigureCanvas(PDViewerCanvas.figure)
        PDViewerCanvas.toolbar = NavigationToolbar(PDViewerCanvas.canvas, self)
        PDViewerCanvas.axes = PDViewerCanvas.figure.add_subplot(1, 1, 1, polar=True)
        PDSeleAll = PDViewerCanvas.axes.scatter([0], [0], edgecolor='k', linewidth=.5, c=[0], s=5,
                                               cmap=cm.hsv)  #empty for init

        # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
        theta_labels = ['±180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']

        PDViewerCanvas.axes.set_ylim(0, 180)
        PDViewerCanvas.axes.set_yticks(np.arange(0, 180, 20))
        PDViewerCanvas.axes.set_xticklabels(theta_labels)
        for tick in PDViewerCanvas.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in PDViewerCanvas.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        PDViewerCanvas.axes.grid(alpha=0.2)

        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        layout.addWidget(PDViewerCanvas.toolbar, 0, 0, 1, 5, QtCore.Qt.AlignVCenter)
        layout.addWidget(PDViewerCanvas.canvas, 1, 0, 10, 5, QtCore.Qt.AlignVCenter)

        self.setLayout(layout)


class TableView(QTableWidget):
    def __init__(self, headers, data, title='', parent=None):
        QTableWidget.__init__(self, parent)
        self.setWindowTitle(title)
        self.BuildTable(headers, data)


    def AddToTable(self, values):
        for k, v in enumerate(values):
            self.AddItem(k, v)


    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QTableWidgetItem(value)
            item = QTableWidgetItem(str(value))
            self.setItem(row, column, item)


    def BuildTable(self, headers, values):
        self.setSortingEnabled(False)
        self.setRowCount(len(values))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.AddToTable(values)
        self.resizeColumnsToContents()
