from PyQt5.QtWidgets import QMainWindow

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
