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

import os, cv2, math
from PySide2 import QtCore, QtWidgets, QtGui, QtUiTools

__copyright__ = 'Copyright © 2020 mmiikeke - All Right Reserved.'

PAGE1 = 'user_interface/form/ui_page1.ui'
PAGE2 = 'user_interface/form/ui_page2_2.ui'
PAGE3 = 'user_interface/form/ui_page3.ui'

class page1(QtCore.QObject):
    def __init__(self, parent=None):

        super(page1, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(PAGE1)
        file.open(QtCore.QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""

class page2(QtCore.QObject):
    def __init__(self, parent=None):

        super(page2, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(PAGE2)
        file.open(QtCore.QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self._widget.progressBar.hide()

        self.set_buttons()
        self.set_grid()

    def set_buttons(self):
        """Setup buttons"""
        self._widget.input_button.clicked.connect(self.select_input)
        self._widget.output_button.clicked.connect(self.select_output)

    def set_grid(self):
        g_layout = QtWidgets.QGridLayout(self._widget.frame_grid)
        g_layout.setSpacing(10)
        g_layout.setMargin(0)
        self._widget.frame_grid.setLayout(g_layout)

    @QtCore.Slot(int)
    def update_progressbar(self, value):
        self._widget.progressBar.setValue(value)

    @QtCore.Slot(str)
    def update_output_imgs(self, path, row, column):
        label = Label(self._widget.frame_grid)
        self._widget.frame_grid.layout().addWidget(label, row, 0, 1, 1)
        self._widget.frame_grid.layout().setAlignment(label, QtCore.Qt.AlignCenter)

        pixmap = QtGui.QPixmap(path)
        label.setPixmap(pixmap, QtCore.QSize(900, 200))
        label.setToolTip(f'image: s_path')

        min_height = self._widget.frame_grid.minimumHeight() + 210
        self._widget.frame_grid.setMinimumHeight(min_height)

        self._widget.scrollArea.verticalScrollBar.setSliderPosition(self._widget.scrollArea.verticalScrollBar.maximum());

    @QtCore.Slot()
    def select_input(self):
        if self._widget.set_video.isChecked():
            file = str(QtWidgets.QFileDialog.getOpenFileName(None, "Select Video", "./", "Video File (*.mp4 *.avi)")[0])
        else:
            file = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))

        self._widget.lineEdit_input.setText(file)
    
    @QtCore.Slot()
    def select_output(self):
        file = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))

        self._widget.lineEdit_output.setText(file)

class page3(QtCore.QObject):
    def __init__(self, parent=None):

        super(page3, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(PAGE3)
        file.open(QtCore.QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""

    def setup_grid(self, p_path, s_path):

        # Read images
        g_layout = QtWidgets.QGridLayout(self._widget.frame_grid)
        g_layout.setSpacing(10)
        g_layout.setMargin(0)

        total_height = 0

        #Set grid content
        for num, path in enumerate(s_path):
            label = Label(self._widget.frame_grid)
            g_layout.addWidget(label, int(num / 2), num % 2, 1, 1)
            g_layout.setAlignment(label, QtCore.Qt.AlignCenter)

            pixmap = QtGui.QPixmap(os.path.join(p_path, path))
            label.setPixmap(pixmap, QtCore.QSize(900, 236))
            label.setToolTip(f'image: s_path')
            
            total_height += (236/2+10)

        self._widget.frame_grid.setMinimumHeight(total_height)

class Label(QtWidgets.QLabel):
    """
    def resizeEvent(self, event):
        if not hasattr(self, 'maximum_size'):
            self.maximum_size = self.size()
        else:
            self.maximum_size = QtCore.QSize(
                max(self.maximum_size.width(), self.width()),
                max(self.maximum_size.height(), self.height()),
            )
        super(Label, self).resizeEvent(event)
    """
    def setPixmap(self, pixmap, size):
        self.maximum_size = size
        scaled = pixmap.scaled(self.maximum_size, QtCore.Qt.KeepAspectRatio)
        super(Label, self).setPixmap(scaled)