#!/usr/bin/python
#
# Copyright © 2020 mmiikeke - All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Add system path
import os, sys
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), 'detection_program'))

from PySide2 import QtWidgets, QtCore, QtGui
from user_interface.main_window import MainWindow

class MyWidget(QtWidgets.QWidget):
    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self, "Close Windoe", "Are you sure you want to exit the application?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if(result == QtWidgets.QMessageBox.Yes):
            sub.thread.do_run = False
            sub.thread.join()
            event.accept()
        else:
            event.ignore()

    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent)
        self.setWindowTitle('Lane Detection based on PINet')
        self.setWindowIcon(QtGui.QIcon('user_interface/media/icon.png'))

    def set_child(self, child):
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.setSpacing(0)
        v_layout.setMargin(0)
        v_layout.addWidget(child)
        self.setLayout(v_layout)
        child.setParent(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    parent = MyWidget()
    sub = MainWindow(parent)
    parent.set_child(sub.window)

    parent.show()
    sys.exit(app.exec_())