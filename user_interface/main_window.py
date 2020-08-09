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

"""Mainwindow of the user interface, host and control the operation.
"""

from collections import OrderedDict

from PySide2 import QtCore
from PySide2.QtCore import Qt, QFile, QPropertyAnimation
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QFont, QIcon, QPixmap

from user_interface.page_widget import page1

__copyright__ = 'Copyright © 2020 mmiikeke - All Right Reserved.'

FORM = 'user_interface/form/ui_main.ui'

IMAGE = 'user_interface/media/bkg.png'

class MainWindow(object):

    def __init__(self, parent=None):
        """Main window, holding all user interface including.

        Args:
          parent: parent class of main window
        Returns:
          None
        Raises:
          None
        """
        self._window = None
        self._pages = OrderedDict()
        self.setup_ui()

    @property
    def window(self):
        """The main window object"""
        return self._window

    def setup_ui(self):
        """Initialize user interface of main window."""
        loader = QUiLoader()
        file = QFile(FORM)
        file.open(QFile.ReadOnly)
        self._window = loader.load(file)
        file.close()

        self._pages['page1'] = page1()

        for index, name in enumerate(self._pages):
            print('pages {} : {} page'.format(index, name))
            self._window.stackedWidget.addWidget(self._pages[name].widget)
        
        self.set_buttons()

    def set_buttons(self):
        """Setup buttons"""
        self._pages['page1'].widget.btn_start.clicked.connect(self.page1_start)

    @QtCore.Slot()
    def page1_start(self):
        print('test')
        print(self._pages['page1'].widget.geometry())

        # ANIMATION
        self.animation = QPropertyAnimation(self._pages['page1'].widget, b"geometry")
        self.animation.setDuration(1000)
        self.animation.setStartValue(self._pages['page1'].widget.geometry())
        self.animation.setEndValue(QtCore.QRect(0, 500, 1000, 435))
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()