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

from PySide2.QtWidgets import QApplication
from PySide2 import QtCore
from user_interface.main_window import MainWindow

if '__main__' == __name__:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)


    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.window.show()

    ret = app.exec_()
    sys.exit(ret)
