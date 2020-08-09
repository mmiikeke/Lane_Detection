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

"""The item widget page
"""
from PySide2 import QtCore
from PySide2.QtCore import Qt, QFile, QObject, QPropertyAnimation
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QFont, QIcon
from PySide2.QtWidgets import QFileDialog

__copyright__ = 'Copyright © 2020 mmiikeke - All Right Reserved.'

PAGE1 = 'user_interface/form/ui_page1.ui'
PAGE2 = 'user_interface/form/ui_page2.ui'

class page1(QObject):
    def __init__(self, parent=None):

        super(page1, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QUiLoader()
        file = QFile(PAGE1)
        file.open(QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""

class page2(QObject):
    def __init__(self, parent=None):

        super(page2, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QUiLoader()
        file = QFile(PAGE2)
        file.open(QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""
        self._widget.input_button.clicked.connect(self.select_input)
        self._widget.output_button.clicked.connect(self.select_output)

    @QtCore.Slot()
    def select_input(self):
        if self._widget.set_video.isChecked():
            file = str(QFileDialog.getOpenFileName(None, "Select Video", "./", "MP4 File (*.mp4)")[0])
        else:
            file = str(QFileDialog.getExistingDirectory(None, "Select Directory"))

        self._widget.lineEdit_input.setText(file)
    
    @QtCore.Slot()
    def select_output(self):
        file = str(QFileDialog.getExistingDirectory(None, "Select Directory"))

        self._widget.lineEdit_output.setText(file)