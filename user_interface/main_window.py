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
import os, threading
from collections import OrderedDict

from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtCore import Qt, QFile, QPropertyAnimation, QParallelAnimationGroup
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QFont, QIcon, QPixmap

from user_interface.page_widget import page1, page2, page3
from detection_program.demo_class import Lane_Detection

__copyright__ = 'Copyright © 2020 mmiikeke - All Right Reserved.'

FORM = 'user_interface/form/ui_main.ui'

IMAGE = 'user_interface/media/bkg.png'

class MainWindow(QtCore.QObject):

    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
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

        self.set_pages()
        self.set_buttons()

    def set_pages(self):
        """Setup pages"""
        self._pages['page1'] = page1()
        self._pages['page2'] = page2()
        self._pages['page3'] = page3()

        # Add to frame
        g_layout = QtWidgets.QGridLayout(self._window.frame_content)
        g_layout.setSpacing(0)
        g_layout.setMargin(0)

        for index, name in enumerate(self._pages):
            print('pages {} : {} page'.format(index, name))
            g_layout.addWidget(self._pages[name].widget, 0, 0, 1, 1)

        self._pages['page2'].widget.stackUnder(self._pages['page1'].widget)
        self._pages['page2'].widget.setDisabled(True)
        self._pages['page2'].widget.hide()
        self._pages['page3'].widget.stackUnder(self._pages['page2'].widget)
        self._pages['page3'].widget.setDisabled(True)
        self._pages['page3'].widget.hide()

    def set_buttons(self):
        """Setup buttons"""
        self._pages['page1'].widget.btn_start.clicked.connect(lambda: self.next_page(self._pages['page1'].widget, self._pages['page2'].widget))
        self._pages['page2'].widget.btn_detect.clicked.connect(lambda: self.detect(self._pages['page2'].widget))
        self._pages['page3'].widget.btn_home.clicked.connect(lambda: self.next_page(self._pages['page3'].widget, self._pages['page1'].widget))

    @QtCore.Slot()
    def next_page(self, a, b):
        a.setDisabled(True)
        b.setGeometry(a.geometry().translated(a.geometry().width() * 1.1, 0))
        b.show()

        print(a.geometry())
        #print(a.parentWidget().layout())

        # ANIMATION
        self.anim_a = QPropertyAnimation(a, b"geometry")
        self.anim_a.setDuration(2000)
        self.anim_a.setStartValue(a.geometry())
        self.anim_a.setEndValue(a.geometry().translated(-a.geometry().width() * 1.5, 0))
        self.anim_a.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        #self.anim_a.start()
        
        print(b.geometry())
        self.anim_b = QPropertyAnimation(b, b"geometry")
        self.anim_b.setDuration(2200)
        self.anim_b.setKeyValueAt(0, a.geometry().translated(a.geometry().width() * 1.1, 0))
        self.anim_b.setKeyValueAt(0.2, a.geometry().translated(a.geometry().width() * 1.1, 0))
        self.anim_b.setKeyValueAt(1, a.geometry())
        self.anim_b.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        #anim_b.start()

        self.group = QParallelAnimationGroup()
        self.group.addAnimation(self.anim_a)
        self.group.addAnimation(self.anim_b)
        self.group.start()

        QtCore.QTimer.singleShot(2000, lambda: self.next_page_callback(a, b))
    
    @QtCore.Slot()
    def next_page_callback(self, a, b):
        b.setDisabled(False)
        a.stackUnder(b)
        a.hide()
    
    @QtCore.Slot()
    def detect(self, widget):
        self.input_path = widget.lineEdit_input.text()
        self.output_path = widget.lineEdit_output.text()
        is_input_video = widget.set_video.isChecked()
        is_output_video = widget.output_video.isChecked()
        is_output_clips = widget.output_clips.isChecked()
        print(f'input:{self.input_path}\noutput:{self.output_path}\nis input video:{is_input_video}\nis output video:{is_output_video}\nis output clips:{is_output_clips}')

        demo = Lane_Detection(self.input_path, self.output_path, is_input_video, is_output_video, is_output_clips, widget)
        demo.detect_callback.connect(self.detect_callback)
        demo.update_progressbar.connect(self._pages['page2'].update_progressbar)
        demo.update_output_imgs.connect(self._pages['page2'].update_output_imgs)
        #demo.run()
        widget.progressBar.show()
        self.thread = threading.Thread(target = demo.run)
        self.thread.do_run = True
        self.thread.start()

    @QtCore.Slot()
    def detect_callback(self, subpaths):
        self.subpaths = subpaths
        #self._pages['page3'].widget.show()
        #self._pages['page3'].setup_grid(os.path.join(self.output_path, 'clips'), self.subpaths)
        #self.next_page(self._pages['page2'].widget, self._pages['page3'].widget)

    @QtCore.Slot()
    def exit(self):
        self._window.close()
