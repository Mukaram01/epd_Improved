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

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCommandLinkButton,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import logging
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
    _DEFAULT_MARGIN = 28
    _DEFAULT_SPACING = 18

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

        self._WINDOW_HEIGHT = 500
        self._WINDOW_WIDTH = 760

        self.setObjectName('epdLauncher')
        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))
        self.setWindowTitle('Easy Perception Deployment')
        self.resize(self._WINDOW_WIDTH, self._WINDOW_HEIGHT)
        self.setMinimumSize(660, 440)

        self._apply_launcher_style()
        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in MainWindow.'''
        self.train_button = self._configure_main_button(
            text='&Train',
            description='Prepare datasets and train a custom perception model.',
            icon_name='train.png',
            tooltip='Train a model (Alt+T)',
            object_name='trainAction',
        )

        self.deploy_button = self._configure_main_button(
            text='&Deploy',
            description='Configure and launch a ROS 2 perception pipeline.',
            icon_name='deploy.png',
            tooltip='Deploy a model (Alt+D)',
            object_name='deployAction',
        )

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setObjectName('quitButton')
        self.quit_button.setIcon(QIcon(self._image_path('quit.png')))
        self.quit_button.setIconSize(QSize(20, 20))
        self.quit_button.setToolTip('Quit the application (Alt+Q)')
        self.quit_button.setShortcut('Alt+Q')
        self.quit_button.setCursor(Qt.PointingHandCursor)
        self.quit_button.setMinimumHeight(38)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
            self._DEFAULT_MARGIN,
        )
        root_layout.setSpacing(self._DEFAULT_SPACING)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        logo_label = QLabel(self)
        logo_label.setObjectName('brandLogo')
        logo_pixmap = QPixmap(self._image_path('epd_desktop.png'))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(
                    56,
                    56,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        logo_label.setFixedSize(60, 60)
        logo_label.setAlignment(Qt.AlignCenter)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 2, 0, 0)
        title_stack.setSpacing(2)

        eyebrow = QLabel('EPD  •  ROS 2 PERCEPTION', self)
        eyebrow.setObjectName('eyebrow')
        title = QLabel('Easy Perception Deployment', self)
        title.setObjectName('launcherTitle')
        subtitle = QLabel(
            'Train vision models or launch a live perception pipeline.', self)
        subtitle.setObjectName('launcherSubtitle')
        subtitle.setWordWrap(True)

        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)

        header_layout.addWidget(logo_label, 0, Qt.AlignTop)
        header_layout.addLayout(title_stack, 1)
        root_layout.addLayout(header_layout)

        section_label = QLabel('Choose a workflow', self)
        section_label.setObjectName('sectionLabel')
        root_layout.addWidget(section_label)

        action_layout = QGridLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setHorizontalSpacing(16)
        action_layout.setVerticalSpacing(16)
        action_layout.addWidget(self.train_button, 0, 0)
        action_layout.addWidget(self.deploy_button, 0, 1)
        action_layout.setColumnStretch(0, 1)
        action_layout.setColumnStretch(1, 1)
        action_layout.setRowStretch(0, 1)
        root_layout.addLayout(action_layout, 1)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 2, 0, 0)
        footer_layout.setSpacing(12)

        footer_label = QLabel(
            'Local GUI  •  configuration changes remain on this workstation', self)
        footer_label.setObjectName('footerLabel')
        footer_layout.addWidget(footer_label, 1)
        footer_layout.addWidget(self.quit_button, 0, Qt.AlignRight)
        root_layout.addLayout(footer_layout)

        self.train_button.clicked.connect(self.openTrainWindow)
        self.deploy_button.clicked.connect(self.deployPackage)
        self.quit_button.clicked.connect(self.closeWindow)
        self._update_button_icon_sizes()

    def _configure_main_button(
            self,
            text,
            description,
            icon_name,
            tooltip,
            object_name):
        button = QCommandLinkButton(text, description, self)
        button.setObjectName(object_name)
        button.setIcon(QIcon(self._image_path(icon_name)))
        button.setToolTip(tooltip)
        button.setMinimumHeight(205)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setCursor(Qt.PointingHandCursor)
        return button

    def _update_button_icon_sizes(self):
        if self.width() < 700:
            top_icon = 58
        elif self.width() < 900:
            top_icon = 72
        else:
            top_icon = 86

        self.train_button.setIconSize(
            self._adaptive_icon_size(self.train_button, top_icon)
        )
        self.deploy_button.setIconSize(
            self._adaptive_icon_size(self.deploy_button, top_icon)
        )

    def _adaptive_icon_size(self, button, preferred):
        button_width = max(button.width() - 48, 32)
        button_height = max(button.height() - 92, 32)
        icon_edge = max(32, min(preferred, button_width, button_height))
        return QSize(icon_edge, icon_edge)

    def _apply_launcher_style(self):
        self.setStyleSheet(
            '''
            QWidget#epdLauncher {
                background-color: #11141a;
                color: #f4f7fb;
            }

            QLabel#brandLogo {
                background-color: #171c24;
                border: 1px solid #2a3240;
                border-radius: 14px;
            }

            QLabel#eyebrow {
                color: #8e9bb0;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#launcherTitle {
                color: #f7f9fc;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#launcherSubtitle {
                color: #aeb8c8;
                font-size: 13px;
            }

            QLabel#sectionLabel {
                color: #dbe2ec;
                font-size: 13px;
                font-weight: 600;
                padding-top: 4px;
            }

            QCommandLinkButton {
                color: #f7f9fc;
                background-color: #191e27;
                border: 1px solid #2b3443;
                border-radius: 16px;
                padding: 22px;
                font-size: 16px;
                font-weight: 600;
                text-align: left;
            }

            QCommandLinkButton:hover {
                background-color: #202735;
                border-color: #596b87;
            }

            QCommandLinkButton:pressed {
                background-color: #161b23;
                border-color: #7f91ad;
            }

            QCommandLinkButton#deployAction {
                background-color: #1a2030;
                border-color: #5367d8;
            }

            QCommandLinkButton#deployAction:hover {
                background-color: #222b43;
                border-color: #7d8cf0;
            }

            QLabel#footerLabel {
                color: #7f8a9c;
                font-size: 11px;
            }

            QPushButton#quitButton {
                color: #b9c2d0;
                background-color: transparent;
                border: 1px solid #303948;
                border-radius: 9px;
                padding: 6px 14px;
                font-size: 12px;
            }

            QPushButton#quitButton:hover {
                color: #ffffff;
                background-color: #242b36;
                border-color: #4c596d;
            }

            QPushButton#quitButton:pressed {
                background-color: #1b212a;
            }
            '''
        )

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
