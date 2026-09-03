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
from windows.deploy_ui_refresh import apply_deploy_ui_refresh
from windows.deploy_ui_scale import apply_deploy_ui_scale


class _WorkflowCard(QPushButton):
    """Clickable launcher card with predictable cross-platform typography."""

    def __init__(
            self,
            title,
            description,
            meta,
            icon_path,
            shortcut,
            object_name,
            parent=None):
        super().__init__(parent)
        self._icon_path = icon_path
        self.setObjectName(object_name)
        self.setToolTip(f'{title} ({shortcut})')
        self.setShortcut(shortcut)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(168)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName('workflowIcon')
        self.icon_label.setFixedSize(66, 66)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label, 0, Qt.AlignTop)

        content = QVBoxLayout()
        content.setContentsMargins(0, 1, 0, 0)
        content.setSpacing(7)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        title_label = QLabel(title, self)
        title_label.setObjectName('workflowTitle')
        title_row.addWidget(title_label)
        title_row.addStretch(1)

        open_label = QLabel('Open  →', self)
        open_label.setObjectName('workflowOpen')
        title_row.addWidget(open_label, 0, Qt.AlignRight)

        description_label = QLabel(description, self)
        description_label.setObjectName('workflowDescription')
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        content.addLayout(title_row)
        content.addWidget(description_label)
        content.addStretch(1)

        meta_label = QLabel(meta, self)
        meta_label.setObjectName('workflowMeta')
        content.addWidget(meta_label)

        layout.addLayout(content, 1)

        for child in self.findChildren(QWidget):
            child.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.set_icon_size(58)

    def set_icon_size(self, edge):
        pixmap = QPixmap(self._icon_path)
        if pixmap.isNull():
            self.icon_label.clear()
            return
        self.icon_label.setPixmap(
            pixmap.scaled(
                edge,
                edge,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class MainWindow(QWidget):
    '''
    The MainWindow class is a PySide6 Graphical User Interface (GUI) window
    that starts up as the first user interface.
    '''
    _MODULE_DIR = Path(__file__).resolve().parent
    _GUI_DIR = _MODULE_DIR.parent
    _DEFAULT_MARGIN = 28
    _DEFAULT_SPACING = 16

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
        apply_deploy_ui_refresh(self.deploy_window)
        apply_deploy_ui_scale(self.deploy_window)
        self.isTrainOpen = False
        self.isDeployOpen = False

        self.train_window.installEventFilter(self)
        self.deploy_window.installEventFilter(self)

        self._WINDOW_HEIGHT = 438
        self._WINDOW_WIDTH = 760

        self.setObjectName('epdLauncher')
        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))
        self.setWindowTitle('Easy Perception Deployment')
        self.resize(self._WINDOW_WIDTH, self._WINDOW_HEIGHT)
        self.setMinimumSize(680, 410)

        self._apply_launcher_style()
        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in MainWindow.'''
        self.train_button = self._configure_main_button(
            title='Train',
            description='Prepare datasets and train a custom perception model.',
            meta='DATASETS  •  LABELS  •  MODEL TRAINING',
            icon_name='train.png',
            shortcut='Alt+T',
            object_name='trainAction',
        )

        self.deploy_button = self._configure_main_button(
            title='Deploy',
            description='Configure and launch a live ROS 2 perception pipeline.',
            meta='MODEL  •  CAMERA  •  DETECTION / TRACKING',
            icon_name='deploy.png',
            shortcut='Alt+D',
            object_name='deployAction',
        )

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setObjectName('quitButton')
        self.quit_button.setIcon(QIcon(self._image_path('quit.png')))
        self.quit_button.setIconSize(QSize(18, 18))
        self.quit_button.setToolTip('Quit the application (Alt+Q)')
        self.quit_button.setShortcut('Alt+Q')
        self.quit_button.setCursor(Qt.PointingHandCursor)
        self.quit_button.setMinimumHeight(36)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(
            self._DEFAULT_MARGIN,
            24,
            self._DEFAULT_MARGIN,
            20,
        )
        root_layout.setSpacing(self._DEFAULT_SPACING)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(14)

        logo_label = QLabel(self)
        logo_label.setObjectName('brandLogo')
        logo_pixmap = QPixmap(self._image_path('epd_desktop.png'))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(
                    50,
                    50,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        logo_label.setFixedSize(54, 54)
        logo_label.setAlignment(Qt.AlignCenter)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(1)

        eyebrow = QLabel('EPD  •  ROS 2 PERCEPTION', self)
        eyebrow.setObjectName('eyebrow')
        title = QLabel('Easy Perception Deployment', self)
        title.setObjectName('launcherTitle')
        subtitle = QLabel(
            'Train vision models or launch a live perception pipeline.', self)
        subtitle.setObjectName('launcherSubtitle')

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
        action_layout.setHorizontalSpacing(14)
        action_layout.addWidget(self.train_button, 0, 0)
        action_layout.addWidget(self.deploy_button, 0, 1)
        action_layout.setColumnStretch(0, 1)
        action_layout.setColumnStretch(1, 1)
        root_layout.addLayout(action_layout)

        root_layout.addStretch(1)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(12)

        footer_label = QLabel(
            'LOCAL GUI  •  CONFIGURATION STAYS ON THIS WORKSTATION', self)
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
            title,
            description,
            meta,
            icon_name,
            shortcut,
            object_name):
        return _WorkflowCard(
            title=title,
            description=description,
            meta=meta,
            icon_path=self._image_path(icon_name),
            shortcut=shortcut,
            object_name=object_name,
            parent=self,
        )

    def _update_button_icon_sizes(self):
        icon_edge = 52 if self.width() < 720 else 58
        self.train_button.set_icon_size(icon_edge)
        self.deploy_button.set_icon_size(icon_edge)

    def _apply_launcher_style(self):
        self.setStyleSheet(
            '''
            QWidget#epdLauncher {
                background-color: #101319;
                color: #f4f7fb;
            }

            QLabel#brandLogo {
                background-color: #171c24;
                border: 1px solid #29313e;
                border-radius: 13px;
            }

            QLabel#eyebrow {
                color: #8490a4;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#launcherTitle {
                color: #f7f9fc;
                font-size: 23px;
                font-weight: 700;
            }

            QLabel#launcherSubtitle {
                color: #a9b3c3;
                font-size: 12px;
            }

            QLabel#sectionLabel {
                color: #dbe2ec;
                font-size: 12px;
                font-weight: 600;
                padding-top: 2px;
            }

            QPushButton#trainAction,
            QPushButton#deployAction {
                background-color: #181d26;
                border: 1px solid #2b3441;
                border-radius: 15px;
                text-align: left;
            }

            QPushButton#trainAction:hover,
            QPushButton#deployAction:hover {
                background-color: #1d2430;
                border-color: #526176;
            }

            QPushButton#trainAction:pressed,
            QPushButton#deployAction:pressed {
                background-color: #151a22;
                border-color: #6b7b94;
            }

            QPushButton#deployAction {
                border-color: #394862;
            }

            QPushButton#deployAction:hover {
                border-color: #667ba0;
            }

            QLabel#workflowIcon {
                background-color: #11151c;
                border: 1px solid #27303c;
                border-radius: 13px;
            }

            QLabel#workflowTitle {
                color: #f6f8fb;
                font-size: 18px;
                font-weight: 700;
            }

            QLabel#workflowDescription {
                color: #b5bfce;
                font-size: 12px;
                font-weight: 400;
            }

            QLabel#workflowMeta {
                color: #778399;
                font-size: 9px;
                font-weight: 700;
            }

            QLabel#workflowOpen {
                color: #8390a5;
                font-size: 11px;
                font-weight: 600;
            }

            QPushButton#trainAction:hover QLabel#workflowOpen,
            QPushButton#deployAction:hover QLabel#workflowOpen {
                color: #c6d2e4;
            }

            QLabel#footerLabel {
                color: #667286;
                font-size: 9px;
                font-weight: 600;
            }

            QPushButton#quitButton {
                color: #aeb8c7;
                background-color: transparent;
                border: 1px solid #2c3543;
                border-radius: 9px;
                padding: 5px 13px;
                font-size: 11px;
            }

            QPushButton#quitButton:hover {
                color: #ffffff;
                background-color: #202731;
                border-color: #4a5769;
            }

            QPushButton#quitButton:pressed {
                background-color: #181e27;
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
