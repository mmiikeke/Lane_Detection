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
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2.QtCore import Qt, QFile, QObject, QPropertyAnimation
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtGui


__copyright__ = 'Copyright © 2020 mmiikeke - All Right Reserved.'

PAGE1 = 'user_interface/form/ui_page1.ui'
PAGE2 = 'user_interface/form/ui_page2.ui'
PAGE3 = 'user_interface/form/ui_page3.ui'

TEMP = 'D:/mike/github/Lane_Detection/data/dd'

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

        self._widget.progressBar.hide()

        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""
        self._widget.input_button.clicked.connect(self.select_input)
        self._widget.output_button.clicked.connect(self.select_output)

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

class page3(QObject):
    def __init__(self, parent=None):

        super(page3, self).__init__(parent)

        self._widget = None

        self.setup_ui()
        
    @property
    def widget(self):
        return self._widget
    
    def setup_ui(self):
        """Initialize user interface of widget."""
        loader = QUiLoader()
        file = QFile(PAGE3)
        file.open(QFile.ReadOnly)
        self._widget = loader.load(file)
        file.close()

        self.set_buttons()
        self.setup_grid()

    def set_buttons(self):
        """Setup buttons"""

    def setup_grid(self):

        # Read images
        paths = os.listdir(TEMP)
        length = len(paths)

        g_layout = QtWidgets.QGridLayout()
        g_layout.setSpacing(0)
        g_layout.setMargin(0)

        #Set grid content
        for num, path in enumerate(paths):
            label = QtWidgets.QLabel(self._widget)
            label.setPixmap(QtGui.QPixmap(os.path.join(TEMP, path)))
            #vlayout = QtWidgets.QVBoxLayout()
            #v_layout.addWidget(label)
            #item = self.create_cell(col_data)
            #item.setToolTip('row{},Col{}'.format(row_num, col_num))
            g_layout.addWidget(label, int(num / 5), num % 5, 1, 1)

        self._widget.frame_grid.setLayout(g_layout)


    def setup_table(self):
        # Read images
        paths = os.listdir(TEMP)
        length = len(paths)

        # Setup TableWidget
        table = self._widget.table_widget
        #table.verticalHeader().setDefaultSectionSize(18)
        #table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)

        #Let whole content fit in the window 
        #table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        table.verticalScrollBar().setVisible(True)
        table.horizontalScrollBar().setVisible(False)

        #Set row & column count
        table.setColumnCount(5)
        table.setRowCount(math.ceil(length / 5))

        #Set table content
        for num, path in enumerate(paths):
            label = QtWidgets.QLabel(self._widget)
            label.setPixmap(QtGui.QPixmap(os.path.join(TEMP, path)))
            #vlayout = QtWidgets.QVBoxLayout()
            #v_layout.addWidget(label)
            #item = self.create_cell(col_data)
            #item.setToolTip('row{},Col{}'.format(row_num, col_num))
            table.setItem(int(num / 5), num % 5, label)

