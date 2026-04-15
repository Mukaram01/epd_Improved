# Copyright 2022 ROS-Industrial Consortium Asia Pacific
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

from PySide6.QtCore import QEvent, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from windows.Deploy import DeployWindow
from windows.Train import TrainWindow


class MainWindow(QWidget):
    '''
    The MainWindow class is a PySide6 Graphical User Interface (GUI) window
    that starts up as the first user interface.
    '''
    _MODULE_DIR = Path(__file__).resolve().parent
    _GUI_DIR = _MODULE_DIR.parent
    _DEFAULT_MARGIN = 10
    _DEFAULT_SPACING = 10

    def __init__(self):
        '''
        The constructor.
        Sets the size of the window.
        Calls setButtons function to populate window with button.
        '''
        super().__init__()

        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%d-%m-%Y-%H-%M-%S")

        log_dir = self._GUI_DIR / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.NOTSET,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=str(log_dir / f'{timestamp_string}.log'),
            filemode='w')
        root_logger = logging.getLogger('')
        console_stream = sys.stderr
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(name)-12s: ' +
            '%(levelname)-8s %(message)s')
        console_handler.setFormatter(formatter)
        if not any(
                type(handler) is logging.StreamHandler
                and handler.stream is console_stream
                for handler in root_logger.handlers):
            root_logger.addHandler(console_handler)
        root_logger.propagate = False

        self.train_window = TrainWindow(False)
        self.deploy_window = DeployWindow(False)
        self.isTrainOpen = False
        self.isDeployOpen = False

        self.train_window.installEventFilter(self)
        self.deploy_window.installEventFilter(self)

        self._WINDOW_HEIGHT = 375
        self._WINDOW_WIDTH = 500

        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))

        self.setWindowTitle('easy_perception_deployment')
        self.setGeometry(0, 0, self._WINDOW_WIDTH, self._WINDOW_HEIGHT)

        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in MainWindow.'''
        self.train_button = self._configure_main_button(
            text='&Train',
            icon_name='train.png',
            tooltip='Train a model (Alt+T)',
            min_height=180,
        )

        self.deploy_button = self._configure_main_button(
            text='&Deploy',
            icon_name='deploy.png',
            tooltip='Deploy a model (Alt+D)',
            min_height=180,
        )

        self.quit_button = self._configure_main_button(
            text='&Quit',
            icon_name='quit.png',
            tooltip='Quit the application (Alt+Q)',
            min_height=100,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
        )
        layout.setSpacing(self._DEFAULT_SPACING)
        top_layout = QGridLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(self._DEFAULT_SPACING)
        top_layout.addWidget(self.train_button, 0, 0)
        top_layout.addWidget(self.deploy_button, 0, 1)
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 1)
        top_layout.setRowStretch(0, 1)
        layout.addLayout(top_layout)
        layout.addWidget(self.quit_button)
        layout.setStretch(0, 3)
        layout.setStretch(1, 2)

        self.train_button.clicked.connect(self.openTrainWindow)
        self.deploy_button.clicked.connect(self.deployPackage)
        self.quit_button.clicked.connect(self.closeWindow)
        self._update_button_icon_sizes()

    def _configure_main_button(self, text, icon_name, tooltip, min_height):
        button = QPushButton(text, self)
        button.setIcon(QIcon(self._image_path(icon_name)))
        button.setToolTip(tooltip)
        button.setMinimumHeight(min_height)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return button

    def _update_button_icon_sizes(self):
        if self.width() < 520:
            top_icon = 60
            quit_icon = 44
        elif self.width() < 760:
            top_icon = 84
            quit_icon = 60
        else:
            top_icon = 108
            quit_icon = 76

        self.train_button.setIconSize(
            self._adaptive_icon_size(self.train_button, top_icon)
        )
        self.deploy_button.setIconSize(
            self._adaptive_icon_size(self.deploy_button, top_icon)
        )
        self.quit_button.setIconSize(
            self._adaptive_icon_size(self.quit_button, quit_icon)
        )

    def _adaptive_icon_size(self, button, preferred):
        button_width = max(button.width() - 30, 32)
        button_height = max(button.height() - 60, 32)
        icon_edge = max(24, min(preferred, button_width, button_height))
        return QSize(icon_edge, icon_edge)

    def deployPackage(self):
        '''A function that is triggered by the button labelled, Deploy.'''
        if self.deploy_window.isVisible():
            self.deploy_window.raise_()
            self.deploy_window.activateWindow()
        else:
            self.deploy_window.show()
            self.deploy_window.raise_()
            self.deploy_window.activateWindow()
        self.isDeployOpen = self.deploy_window.isVisible()

    def openTrainWindow(self):
        '''A function that is triggered by the button labelled, Train.'''
        if self.train_window.isVisible():
            self.train_window.raise_()
            self.train_window.activateWindow()
        else:
            self.train_window.show()
            self.train_window.raise_()
            self.train_window.activateWindow()
        self.isTrainOpen = self.train_window.isVisible()

    def eventFilter(self, obj, event):
        close_hide_show = (QEvent.Close, QEvent.Hide, QEvent.Show)
        if obj is self.train_window and event.type() in close_hide_show:
            self.isTrainOpen = self.train_window.isVisible()
        elif obj is self.deploy_window and event.type() in close_hide_show:
            self.isDeployOpen = self.deploy_window.isVisible()
        return super().eventFilter(obj, event)

    def closeWindow(self):
        '''A function that is triggered by the button labelled, Quit.'''
        self.close()
        self.train_window.close()
        self.deploy_window.close()

    def closeEvent(self, event):
        self.deploy_window.shutdown()
        self.train_window.close()
        event.accept()

    def resizeEvent(self, event):
        self._update_button_icon_sizes()
        super().resizeEvent(event)

    def _image_path(self, image_name):
        return str(self._GUI_DIR / 'img' / image_name)
