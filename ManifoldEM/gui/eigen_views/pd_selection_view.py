import os
import numpy as np

import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QDialog, QLabel, QFrame, QPushButton, QTabWidget,
                             QLayout, QGridLayout, QProgressBar, QDesktopWidget, QTableWidget, QTableWidgetItem)

from ManifoldEM.data_store import data_store
from ManifoldEM.params import p


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

        self.progBar1 = QProgressBar(self)  #minimum=0,maximum=1,value=0)
        self.progBar1.setRange(0, 100)
        self.progBar1.setVisible(False)
        self.progBar1.setValue(0)

        self.progBar2 = QProgressBar(self)  #minimum=0,maximum=1,value=0)
        self.progBar2.setRange(0, 100)
        self.progBar2.setVisible(False)
        self.progBar2.setValue(0)

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
        self.btn_anchSave.clicked.connect(self.anchorSave)
        self.btn_anchLoad = QPushButton('Load Anchors')
        self.btn_anchLoad.clicked.connect(self.anchorLoad)

        self.btn_trashList = QPushButton('List Removals')
        self.btn_trashList.clicked.connect(self.trashList)
        self.btn_trashReset = QPushButton('Reset Removals')
        self.btn_trashReset.clicked.connect(self.trashReset)
        self.btn_trashSave = QPushButton('Save Removals')
        self.btn_trashSave.clicked.connect(self.trashSave)
        self.btn_trashLoad = QPushButton('Load Removals')
        self.btn_trashLoad.clicked.connect(self.trashLoad)

        self.btn_occList = QPushButton('List Occupancies')
        self.btn_occList.clicked.connect(self.occListGen)
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
        layout.addWidget(self.progBar1, 1, 0, 1, 6)

        layout.addWidget(label_edgeLarge2, 2, 0, 1, 6, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_trash, 2, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashList, 2, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashReset, 2, 2, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashSave, 2, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashLoad, 2, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.progBar2, 3, 0, 1, 6)

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

        elif reply == QMessageBox.No:
            pass


    def anchorSave(self):
        temp_anch_list = []
        anch_sum = 0
        for i in range(1, P3.PrD_total + 1):
            if P4.anchorsAll[i].isChecked():
                anch_sum += 1
        if anch_sum == 0:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            msg = 'At least one anchor must first be selected before saving.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        elif anch_sum > 0:
            box = QMessageBox(self)
            self.setWindowTitle('ManifoldEM Save Data')
            box.setText('<b>Save Current Anchors</b>')
            box.setIcon(QMessageBox.Information)
            msg = 'Performing this action will save a list of all active anchors\
                    to the <i>outputs/CC</i> directory for future reference.\
                    <br /><br />\
                    Do you want to proceed?'

            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()

            if reply == QMessageBox.Yes:

                PrDs = []
                CC1s = []
                S1s = []
                CC2s = []
                S2s = []
                colors = []
                P4.anch_list = []

                idx = 0
                for i in range(1, P3.PrD_total + 1):
                    if P4.anchorsAll[i].isChecked():
                        PrDs.append(int(i))
                        # CC1s:
                        CC1s.append(int(P4.reactCoord1All[i].value()))
                        # S1s:
                        if P4.senses1All[i].currentText() == 'S1: FWD':
                            S1s.append(int(1))
                        else:
                            S1s.append(int(-1))
                        # CC2s:
                        CC2s.append(int(P4.reactCoord2All[i].value()))
                        # S2s:
                        if P4.senses2All[i].currentText() == 'S2: FWD':
                            S2s.append(int(1))
                        else:
                            S2s.append(int(-1))
                        # colors:
                        colors.append(P3.col[int(i - 1)])
                        idx += 1

                if P3.user_dimensions == 1:
                    temp_anch_list = zip(PrDs, CC1s, S1s, colors)
                elif P3.user_dimensions == 2:
                    temp_anch_list = zip(PrDs, CC1s, S1s, CC2s, S2s, colors)

                timestr = time.strftime("%Y%m%d-%H%M%S")
                tempAnchInputs = os.path.join(p.CC_dir, f'temp_anchors_{timestr}.txt')

                np.savetxt(tempAnchInputs, list(temp_anch_list), fmt='%i', delimiter='\t')

                box = QMessageBox(self)
                box.setWindowTitle('ManifoldEM Save Current Anchors')
                box.setIcon(QMessageBox.Information)
                box.setText('<b>Saving Complete</b>')
                msg = 'Current anchor selections have been saved to the <i>outputs/CC</i> directory.'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

            elif reply == QMessageBox.No:
                pass

    def anchorLoad(self):
        self.btn_anchList.setDisabled(True)
        self.btn_anchReset.setDisabled(True)
        self.btn_anchSave.setDisabled(True)
        self.btn_anchLoad.setDisabled(True)
        self.btn_trashList.setDisabled(True)
        self.btn_trashReset.setDisabled(True)
        self.btn_trashSave.setDisabled(True)
        self.btn_trashLoad.setDisabled(True)
        self.btn_occList.setDisabled(True)
        self.btn_rebedList.setDisabled(True)

        anch_sum = 0
        for i in range(1, P3.PrD_total + 1):
            if P4.anchorsAll[i].isChecked():
                anch_sum += 1
        if anch_sum == 0:
            self.fname = QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                                    ('Data Files (*.txt)'))[0]
            if self.fname:
                try:
                    if P3.user_dimensions == 1:
                        data = []
                        with open(self.fname) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        PrDs = data[0]
                        CC1s = data[1]
                        S1s = data[2]

                        data_all = np.column_stack((PrDs, CC1s, S1s))
                        PrD = []
                        CC1 = []
                        S1 = []
                        Color = []
                        idx = 0
                        for i, j, k in data_all:
                            PrD.append(int(i))
                            CC1.append(int(j))
                            S1.append(int(k))
                            idx += 1

                        P4.anch_list = zip(PrD, CC1, S1)
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip)  #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                        idx = 0
                        prog = 0
                        self.progBar1.setValue(prog)
                        self.progBar1.setVisible(True)

                        for i in PrD:
                            P4.entry_PrD.setValue(int(i))
                            P4.user_PrD = i
                            P4.PrD_hist = i
                            if P4.trashAll[i].isChecked() == False:  #avoids conflict
                                P4.anchorsAll[i].setChecked(True)
                            P4.reactCoord1All[i].setValue(CC1[idx])
                            if S1[idx] == 1:
                                P4.senses1All[i].setCurrentIndex(0)
                            elif S1[idx] == -1:
                                P4.senses1All[i].setCurrentIndex(1)
                            prog += (1. / len(PrD)) * 100
                            self.progBar1.setValue(prog)
                            idx += 1

                        P4.entry_PrD.setValue(1)

                    elif P3.user_dimensions == 2:
                        fname = os.path.join(p.CC_dir, 'user_anchors.txt')
                        data = []
                        with open(fname) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        PrDs = data[0]
                        CC1s = data[1]
                        S1s = data[2]
                        CC2s = data[3]
                        S2s = data[4]

                        data_all = np.column_stack((PrDs, CC1s, S1s, CC2s, S2s))
                        PrD = []
                        CC1 = []
                        S1 = []
                        CC2 = []
                        S2 = []
                        idx = 0
                        for i, j, k, l, m in data_all:
                            PrD.append(int(i))
                            CC1.append(int(j))
                            S1.append(int(k))
                            CC2.append(int(l))
                            S2.append(int(m))

                        P4.anch_list = zip(PrD, CC1, S1, CC2, S2)
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip)  #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                        idx = 0
                        prog = 0
                        self.progBar1.setValue(prog)
                        self.progBar1.setVisible(True)
                        for i in PrD:
                            P4.user_PrD = i
                            P4.PrD_hist = i
                            if P4.trashAll[i].isChecked() == False:  #avoids conflict
                                P4.anchorsAll[i].setChecked(True)
                            P4.reactCoord1All[i].setValue(CC1[idx])
                            P4.reactCoord2All[i].setValue(CC2[idx])
                            if S1[idx] == 1:
                                P4.senses1All[i].setCurrentIndex(0)
                            elif S1[idx] == -1:
                                P4.senses1All[i].setCurrentIndex(1)
                            if S2[idx] == 1:
                                P4.senses2All[i].setCurrentIndex(0)
                            elif S2[idx] == -1:
                                P4.senses2All[i].setCurrentIndex(1)
                            prog += (1. / len(PrD)) * 100
                            self.progBar1.setValue(prog)
                            idx += 1
                        P4.entry_PrD.setValue(1)

                    self.progBar1.setValue(100)

                    box = QMessageBox(self)
                    box.setWindowTitle('ManifoldEM Load Previous Anchors')
                    box.setIcon(QMessageBox.Information)
                    box.setText('<b>Loading Complete</b>')
                    msg = 'Previous anchor selections have been loaded on the <i>Eigenvectors</i> tab.'
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setInformativeText(msg)
                    reply = box.exec_()

                except:
                    box = QMessageBox(self)
                    box.setWindowTitle('ManifoldEM Error')
                    box.setText('<b>Input Error</b>')
                    box.setIcon(QMessageBox.Warning)
                    box.setInformativeText('Incorrect file structure detected.')
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setDefaultButton(QMessageBox.Ok)
                    ret = box.exec_()

                    self.progBar1.setVisible(False)
                    self.progBar1.setValue(0)
            else:
                pass

        elif anch_sum > 0:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            msg = 'To load anchors from a previous session, first clear all currently selected\
                    anchors via the <i>Reset Anchors</i> button.'

            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()

        self.btn_anchList.setDisabled(False)
        self.btn_anchReset.setDisabled(False)
        self.btn_anchSave.setDisabled(False)
        self.btn_anchLoad.setDisabled(False)
        self.btn_trashList.setDisabled(False)
        self.btn_trashReset.setDisabled(False)
        self.btn_trashSave.setDisabled(False)
        self.btn_trashLoad.setDisabled(False)
        self.btn_occList.setDisabled(False)
        self.btn_rebedList.setDisabled(False)

    def trashList(self):
        PrDs = []
        trashed = []

        trash_sum = 0
        for i in range(1, P3.PrD_total + 1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1

        if trash_sum == 0:
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
            ret = box.exec_()

        else:
            for i in range(1, P3.PrD_total + 1):
                if P4.trashAll[i].isChecked():
                    trashed.append('True')
                else:
                    trashed.append('False')
                PrDs.append(int(i))

            sorted_trash = sorted(zip(PrDs, trashed), key=lambda x: x[1], reverse=True)
            self.anchor_table = trashTable(data=sorted_trash)
            sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
            self.anchor_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
            self.anchor_table.show()

    # reset assignments of all PD removals:
    def trashReset(self):
        box = QMessageBox(self)
        self.setWindowTitle('Reset PD Removals')
        box.setText('<b>Reset Warning</b>')
        box.setIcon(QMessageBox.Warning)
        msg = 'Performing this action will deselect all active removals.\
                <br /><br />\
                Do you want to proceed?'

        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setInformativeText(msg)
        reply = box.exec_()

        if reply == QMessageBox.Yes:
            for i in range(1, P3.PrD_total + 1):
                if P4.trashAll[i].isChecked():
                    P4.trashAll[i].setChecked(False)
                    P4.anchorsAll[i].setDisabled(False)

            self.progBar2.setVisible(False)
            self.progBar2.setValue(0)

            P1.x4 = []
            P1.y4 = []
            P1.z4 = []
            P1.a4 = []
            P4.viz2.update_scene3()

        elif reply == QMessageBox.No:
            pass

    def trashSave(self):
        trash_sum = 0
        for i in range(1, P3.PrD_total + 1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1
        if trash_sum == 0:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            msg = 'At least one PD must first be selected for removal before saving.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        elif trash_sum > 0:
            box = QMessageBox(self)
            self.setWindowTitle('ManifoldEM Save Data')
            box.setText('<b>Save Current Removals</b>')
            box.setIcon(QMessageBox.Information)
            msg = 'Performing this action will save a list of all PDs set for removal\
                    to the <i>outputs/CC</i> directory for future reference.\
                    <br /><br />\
                    Do you want to proceed?'

            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()

            if reply == QMessageBox.Yes:

                trashList = []

                for i in range(1, P3.PrD_total + 1):
                    if P4.trashAll[i].isChecked():
                        trashList.append(int(1))
                    else:
                        trashList.append(int(0))

                timestr = time.strftime("%Y%m%d-%H%M%S")
                trashDir = os.path.join(p.CC_dir, f'temp_removals_{timestr}.txt')
                np.savetxt(trashDir, trashList, fmt='%i', delimiter='\t')

                box = QMessageBox(self)
                box.setWindowTitle('ManifoldEM Save Current Removals')
                box.setIcon(QMessageBox.Information)
                box.setText('<b>Saving Complete</b>')
                msg = 'Current removal selections have been saved to the <i>outputs/CC</i> directory.'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

            elif reply == QMessageBox.No:
                pass


    def trashLoad(self):
        self.btn_anchList.setDisabled(True)
        self.btn_anchReset.setDisabled(True)
        self.btn_anchSave.setDisabled(True)
        self.btn_anchLoad.setDisabled(True)
        self.btn_trashList.setDisabled(True)
        self.btn_trashReset.setDisabled(True)
        self.btn_trashSave.setDisabled(True)
        self.btn_trashLoad.setDisabled(True)
        self.btn_occList.setDisabled(True)
        self.btn_rebedList.setDisabled(True)

        P4.trash_list = []
        trash_sum = 0
        for i in range(1, P3.PrD_total + 1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1
        if trash_sum == 0:
            fname = QFileDialog.getOpenFileName(self, 'Choose Data File', '', ('Data Files (*.txt)'))[0]
            if fname:
                try:
                    data = []
                    with open(fname) as values:
                        for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                            data.append(column)
                    P4.trash_list = data[0]
                    p.set_trash_list(P4.trash_list)

                    trashLen = 0
                    for i in P4.trash_list:
                        if int(i) == int(1):
                            trashLen += 1

                    idx = 1  #PD index
                    prog = 0
                    self.progBar2.setValue(prog)
                    self.progBar2.setVisible(True)
                    for i in P4.trash_list:
                        if int(i) == int(1):  #if PD set to True (remove)
                            P4.entry_PrD.setValue(idx)
                            P4.user_PrD = idx
                            P4.PrD_hist = idx
                            P4.trashAll[idx].setChecked(True)
                            P4.anchorsAll[idx].setChecked(False)
                        prog += (1. / trashLen) * 100
                        self.progBar2.setValue(prog)
                        idx += 1

                    P4.entry_PrD.setValue(1)
                    self.progBar2.setValue(100)

                    box = QMessageBox(self)
                    box.setWindowTitle('ManifoldEM Load Previous Removals')
                    box.setIcon(QMessageBox.Information)
                    box.setText('<b>Loading Complete</b>')
                    msg = 'Previous removal selections have been loaded on the <i>Eigenvectors</i> tab.'
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setInformativeText(msg)
                    reply = box.exec_()

                except:  #IndexError:
                    box = QMessageBox(self)
                    box.setWindowTitle('ManifoldEM Error')
                    box.setText('<b>Input Error</b>')
                    box.setIcon(QMessageBox.Warning)
                    box.setInformativeText('Incorrect file structure detected.')
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setDefaultButton(QMessageBox.Ok)
                    ret = box.exec_()
            else:
                pass

        elif trash_sum > 0:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Warning')
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            msg = 'To load PD removals from a previous session, first clear all currently selected\
                    PD removals via the <i>Reset Removals</i> button.'

            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()

        self.btn_anchList.setDisabled(False)
        self.btn_anchReset.setDisabled(False)
        self.btn_anchSave.setDisabled(False)
        self.btn_anchLoad.setDisabled(False)
        self.btn_trashList.setDisabled(False)
        self.btn_trashReset.setDisabled(False)
        self.btn_trashSave.setDisabled(False)
        self.btn_trashLoad.setDisabled(False)
        self.btn_occList.setDisabled(False)
        self.btn_rebedList.setDisabled(False)

    def occListGen(self):
        sorted_PrDs = sorted(zip(P1.thresh_PrDs, P1.thresh_occ), key=lambda x: x[1], reverse=True)
        self.PrD_table = occTable(data=sorted_PrDs)
        sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
        self.PrD_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
        self.PrD_table.show()

    def view_reembeddings_table(self):
        # read points from re-embedding file:
        fname = os.path.join(p.euler_dir, 'PrD_embeds.txt')
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)

        rebeds0 = data[0]
        rebeds = []
        total = []
        idx = 1
        for i in rebeds0:
            if int(i) == 0:
                rebeds.append('True')
            else:
                rebeds.append('False')
            total.append(idx)
            idx += 1

        if len(rebeds) > 0:
            sorted_rebeds = sorted(zip(total, rebeds), key=lambda x: x[1], reverse=True)
            self.rebed_table = TableView(['PD', 'Re-embedded'], sorted_rebeds, title='PD Re-embeddings')
            sizeObject = QDesktopWidget().screenGeometry(-1)  #user screen size
            self.rebed_table.move((sizeObject.width() // 2) - 100, (sizeObject.height() // 2) - 300)
            self.rebed_table.show()
        else:
            box = QMessageBox(self)
            box.setWindowTitle('ManifoldEM Error')
            box.setText('<b>Input Error</b>')
            box.setIcon(QMessageBox.Information)
            box.setInformativeText('No manifold re-embeddings have been performed.\
                                    <br /><br />\
                                    Manifolds for each PD can be individually re-embedded\
                                    within the <i>View Chosen Topos</i> window.')
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.exec_()


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
