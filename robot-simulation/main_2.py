import sys
from PyQt5.QtCore import Qt, QEvent
import numpy as np
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QApplication, QMainWindow
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl

# External files needed:

from Robot import Robot # OpenGL Robot object
from Robot_2 import Robot_2
"""
Simulation of a 6 DOF robot.
- Cédric Wassenaar & Joeri Verkerk
- C.M.Wassenaar@student.hhs.nl
- 24-09-2018

DEPENDENCIES:
- DOF3_Robot.py
- qtframeV3.py
- rm_utilitiesV3.py
- PyQt5

For:
- kinematics see rm_utilitiesV3 & DOF3_Robot
- PyQt5 implementation see qtframeV3 & main
- 3D rendering see class RenderWindow
"""

glo_position = [3,0,1,0,-1.57079, 0]
class RenderWindow(QWidget):
    """Class to render a 3D view, inherits QWidget"""
    def __init__(self, parent_class):
        QWidget.__init__(self)
        self.vbox = QVBoxLayout(self)
        self.parent_class = parent_class
        self.view3D = gl.GLViewWidget()
        self.fps = 60
        self.view3D.opts['elevation'] = 45
        self.view3D.opts['azimuth'] = -45
        self.view3D.opts['fov'] = 2
        self.view3D.setCameraPosition(distance=500)
        self.grid0 = gl.GLGridItem()
        self.grid0.setSize(12,12,1)
        self.view3D.addItem(self.grid0)
        self.vbox.addWidget(self.view3D)
        self.setLayout(self.vbox)
        self.timer = QTimer()
        # self.installEventFilter(self)  # 安装事件过滤器
        # Robot declaration using the DOF3_Robot
        self.r1 = Robot(self.view3D)

        # Sets the robot trajectory as: array(X, Y, Z, Ax, Ay, Az) (in radians)
        self.r1.set_new_trajectory(np.array([[1.5, 0, 2.5, 0, -1.57079, 0],
                                             [1.5,0,1.8,0,-1.57079,0],
                                   [1.5, 0, 1, 0, -1.57079, 0],
                                   [1.5, 0, 2.5, 0,-1.57079,0],
                                   [1.5, 1, 2.5, 0, -1.57079, 0],
                                   [1.5, 1, 1.8, 0,-1.57079,0],
                                   [1.5, 0, 1.8, 0, -1.57079, 0],
                                   [1.5, 0, 2.5, 0, -1.57079, 0]
                                   ]), 30)
        self.r1.link[0].frame.translate(-2, -2, 0)
        self.r2 = Robot(self.view3D)
        self.r2.set_new_trajectory(np.array([[1.5, 0, 1, 0, -1.57079, 0],
                                   [1.5, 0, 2.5, 0, -1.57079, 0],
                                   [1.5, 0, 1, 0, -1.57079, 0],
                                   [1.5, 1, 1, 0, -1.57079, 0],
                                   [1.5, 1, 2.5, 0, -1.57079, 0],
                                   [1.5, 0, 2.5, 0, -1.57079, 0],
                                   [1.5, 0, 1, 0, -1.57079, 0],
                                   ]), 30)
        self.r2.link[0].frame.translate(-2, 0, 0)
        self.r3 = Robot(self.view3D)
        self.r3.set_new_trajectory(np.array([[1.5, 0, 2.5, 0, -1.57079, 0],
                                   [1.5, 0, 1.8, 0, -1.57079, 0],
                                   [1.5, 1, 1.8, 1.57079, -1.57079, 0],
                                   [1.5, 1, 2.5, 0, -1.57079, 0],
                                   # [1.5, 1, 1, 0, -1.57079, 0],
                                   #           [1.5, 0, 1, 0, -1.57079, 0],
                                             [1.5, 1, 1, 0, -1.57079, 0],
                                             [1.5, 1, 1.8, 0, -1.57079, 0],
                                             [1.5, 0, 1.8, 0, -1.57079, 0],
                                             [1.5, 0, 2.5, 0, -1.57079, 0]
                                   ]), 30)
        self.r3.link[0].frame.translate(-2, 2, 0)
        # Start frame update timer
        self.start()
        self.r4 = Robot_2(self.view3D)
        self.r4.link[0].frame.translate(2, 0, 0)
        # self.r4.position = [-2, 1, 1.8, 0, -1.57079, 0]
        self.destroyed.connect(self._on_destroyed)

    def update_window(self):
        """Render new frame of 3D view"""
        # Update robot position
        self.r1.update_window()
        self.r2.update_window()
        self.r3.update_window()
        self.r4.update_window_new()
        # self.r4.update_window()
        # now update the OpenGL graphics in window
        self.view3D.update()

    def start(self):
        """"Starts timer to call window update"""
        self.timer.timeout.connect(self.update_window)
        self.timer.start(int(self.fps*0.001))

    def update_timer(self, fps):
        self.fps = fps
        self.timer.start(int(self.fps*0.001))

    @staticmethod
    def _on_destroyed(self):
        """Dubbel check to kill timer"""
        if hasattr(self, 'timer'):
            self.timer.stop()


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("鼠标键盘事件示例")
        self.setCentralWidget(RenderWindow(None))  # 指定主窗口中心部件
        self.statusBar().showMessage("ready")  # 状态栏显示信息
        self.resize(1200, 1085)
        self.posi = [0,1]
    def keyPressEvent(self, event):
        from main_2 import glo_position
        key = event.key()

        if key == Qt.Key_W:
            glo_position[0] += 0.05
        if key == Qt.Key_S:
            glo_position[0] -= 0.05
        if key == Qt.Key_A:
            glo_position[1] += 0.05
        if key == Qt.Key_D:
            glo_position[1] -= 0.05
        if key == Qt.Key_Up:
            glo_position[2] += 0.05
        if key == Qt.Key_Down:
            glo_position[2] -= 0.05
        if key == Qt.Key_1:
            glo_position[3] += 0.04
        if key == Qt.Key_2:
            glo_position[3] -= 0.04
        if key == Qt.Key_3:
            glo_position[4] += 0.04
        if key == Qt.Key_4:
            glo_position[4] -= 0.04
        if key == Qt.Key_5:
            glo_position[5] += 0.04
        if key == Qt.Key_6:
            glo_position[5] -= 0.04

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())