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

from user_interface.page_widget import page1, page2

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

        #self._window.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        #self._window.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        #self.ui.frame_label_top_btns.mouseDoubleClickEvent = dobleClickMaximizeRestore

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
        self._pages['page2'] = page2()

        for index, name in enumerate(self._pages):
            print('pages {} : {} page'.format(index, name))
            self._window.stackedWidget.addWidget(self._pages[name].widget)
        
        self._window.stackedWidget.setCurrentIndex(0)

        self.set_buttons()

        #self._window.stackedWidget.setLayout(g_layout)
        #self._window.frame_content.addWidget(self._pages[name].widget)

        #self._window.widget_stack.setCurrentIndex(0)

        # Build up signal / slot
        #self._option_panel.currentItemChanged.connect(self.set_page)

    def set_buttons(self):
        """Setup buttons"""
        self._pages['page1'].widget.btn_start.clicked.connect(lambda: self.next_page(1))

        #self._window.btn_page1_start.clicked.connect(self.page1_start)
        #self._window.import_btn.setIcon(QIcon('basic_widget\\media\\import.svg'))
        #self._window.btn_toggle.clicked.connect(lambda: self.toggleMenu(250, True))

        #self._window.btn_menu_1.clicked.connect(lambda: self.set_page(0))
        #self._window.btn_menu_2.clicked.connect(lambda: self.set_page(1))
        #self._window.btn_menu_3.clicked.connect(lambda: self.set_page(2))
        
        #p = QPixmap(IMAGE);
        #self._window.label_page1_bg.setPixmap(p)
        #self._window.label_page1_bg.setMask(p.mask());
        #w = self._window.frame_content.width();
        #h = self._window.frame_content.height();
        #print(f'w={w}, h={h}')
        # set a scaled pixmap to a w x h window keeping its aspect ratio 
        #self._window.label_page1_bg.setPixmap(p.scaled(1500,1200,QtCore.Qt.KeepAspectRatio));

    @QtCore.Slot()
    def next_page(self, page):
        
        current_widget = self._window.stackedWidget.currentWidget()
        print(current_widget.geometry())

        # ANIMATION
        self.animation = QPropertyAnimation(current_widget, b"geometry")
        self.animation.setDuration(2000)
        self.animation.setStartValue(current_widget.geometry())
        self.animation.setEndValue(QtCore.QRect(-1000, 0, 1000, 435))
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()

        QtCore.QTimer.singleShot(2000, lambda: self.set_page(1))

    @QtCore.Slot()
    def set_page(self, page):
        """Slot, switch page of stack widget"""
        self._window.stackedWidget.setCurrentIndex(page)

                # ANIMATION
        self.animation = QPropertyAnimation(self._window.stackedWidget.currentWidget(), b"geometry")
        self.animation.setDuration(2000)
        self.animation.setStartValue(QtCore.QRect(1000, 0, 1000, 435))
        self.animation.setEndValue(QtCore.QRect(0, 0, 1000, 435))
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()

    @QtCore.Slot()
    def toggleMenu(self, maxWidth, enable):
        print('start')
        if enable:

            # GET WIDTH
            width = self._window.frame_left_menu.width()
            print(f'width = {width}')
            maxExtend = maxWidth
            standard = 70

            # SET MAX WIDTH
            if width == 70:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # ANIMATION
            self.animation = QPropertyAnimation(self._window.frame_left_menu, b"minimumWidth")
            self.animation.setDuration(400)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
            self.animation.start()
    
    @QtCore.Slot(str)
    def say_hello(self, msg):
        print('Hello ' + msg)

    @QtCore.Slot()
    def exit(self):
        self._window.close()
