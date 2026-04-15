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
from PySide6.QtWidgets import (QGridLayout, QLabel, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

import logging
import os
import subprocess
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
        self._preflight_passed = False

        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))

        self.setWindowTitle('easy_perception_deployment')
        self.setGeometry(0, 0, self._WINDOW_WIDTH, self._WINDOW_HEIGHT)

        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in MainWindow.'''
        self.train_button = QPushButton('Train', self)
        self.train_button.setIcon(QIcon(self._image_path('train.png')))
        self.train_button.setIconSize(QSize(100, 100))
        self.train_button.setFixedHeight(250)

        self.deploy_button = QPushButton('Deploy', self)
        self.deploy_button.setIcon(QIcon(self._image_path('deploy.png')))
        self.deploy_button.setIconSize(QSize(100, 100))
        self.deploy_button.setFixedHeight(250)

        self.preflight_button = QPushButton('Preflight Check', self)
        self.preflight_button.setFixedHeight(60)

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setIcon(QIcon(self._image_path('quit.png')))
        self.quit_button.setIconSize(QSize(250, 250))
        self.quit_button.setFixedHeight(125)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top_layout = QGridLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(self.train_button, 0, 0)
        top_layout.addWidget(self.deploy_button, 0, 1)
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 1)
        layout.addLayout(top_layout)
        layout.addWidget(self.preflight_button)
        self.preflight_status_label = QLabel('Preflight required before Deploy/Train.')
        self.preflight_status_label.setWordWrap(True)
        layout.addWidget(self.preflight_status_label)
        layout.addWidget(self.quit_button)

        self.train_button.clicked.connect(self.openTrainWindow)
        self.deploy_button.clicked.connect(self.deployPackage)
        self.preflight_button.clicked.connect(self.runPreflightChecks)
        self.quit_button.clicked.connect(self.closeWindow)
        self._set_preflight_gate(False)

    def _set_preflight_gate(self, passed):
        self._preflight_passed = passed
        self.train_button.setEnabled(passed)
        self.deploy_button.setEnabled(passed)

    def runPreflightChecks(self):
        checks = []

        ros_cli = self._run_cmd(['ros2', '--help'], timeout=4)
        ros_setup_ok = ros_cli['ok']
        checks.append({
            'name': 'ROS setup availability',
            'ok': ros_setup_ok,
            'detail': ('ros2 CLI reachable.'
                       if ros_setup_ok else
                       'ros2 command unavailable. Run: source /opt/ros/humble/setup.bash')
        })

        workspace_ok, workspace_detail = self._validate_workspace_setup_path()
        checks.append({
            'name': 'Workspace install/setup path',
            'ok': workspace_ok,
            'detail': workspace_detail
        })

        model_dir = self._GUI_DIR.parent / 'data' / 'model'
        model_files = list(model_dir.glob('*.onnx')) if model_dir.exists() else []
        model_ok = len(model_files) > 0
        checks.append({
            'name': 'Model files under data/model',
            'ok': model_ok,
            'detail': (f'Found {len(model_files)} ONNX model(s) in {model_dir}.'
                       if model_ok else
                       f'No ONNX models found in {model_dir}. Run: bash scripts/download_models.sh')
        })

        docker_info = self._run_cmd(['docker', 'info'], timeout=6)
        docker_ok = docker_info['ok']
        checks.append({
            'name': 'Docker accessibility',
            'ok': docker_ok,
            'detail': ('Docker daemon reachable.'
                       if docker_ok else
                       'Docker unavailable. Run: sudo systemctl start docker && sudo usermod -aG docker $USER')
        })

        topic_list = self._run_cmd(['ros2', 'topic', 'list', '-t'], timeout=6)
        topic_ok = topic_list['ok']
        checks.append({
            'name': 'Image topic discovery health (ros2 topic list -t)',
            'ok': topic_ok,
            'detail': ('Topic discovery command succeeded.'
                       if topic_ok else
                       'Topic discovery failed. Run: source /opt/ros/humble/setup.bash && source <workspace>/install/setup.bash')
        })

        all_ok = all(check['ok'] for check in checks)
        self._set_preflight_gate(all_ok)

        result_lines = []
        for check in checks:
            marker = '✓' if check['ok'] else '✗'
            result_lines.append(f"{marker} {check['name']}: {check['detail']}")
        summary = 'Preflight passed. Deploy/Train enabled.' if all_ok else (
            'Preflight failed. Resolve failed checks and re-run.')
        result_lines.append('')
        result_lines.append(summary)
        combined = '\n'.join(result_lines)
        self.preflight_status_label.setText(combined)
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Preflight Check')
        msg_box.setText(combined)
        msg_box.exec()

    def _run_cmd(self, cmd, timeout=4):
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {'ok': False}
        return {'ok': completed.returncode == 0}

    def _validate_workspace_setup_path(self):
        workspace_env = os.environ.get('EPD_WS')
        candidates = []
        if workspace_env:
            candidates.append(Path(workspace_env) / 'install' / 'setup.bash')
        candidates.extend([
            Path.home() / 'epd_ros2_ws' / 'install' / 'setup.bash',
            self._PACKAGE_ROOT() / 'install' / 'setup.bash',
        ])
        for candidate in candidates:
            if candidate.exists():
                return True, f'Found setup.bash: {candidate}'
        return False, (
            'Could not find install/setup.bash. Run colcon build then '
            'source <workspace>/install/setup.bash')

    def _PACKAGE_ROOT(self):
        return self._GUI_DIR.parents[2]

    def deployPackage(self):
        '''A function that is triggered by the button labelled, Deploy.'''
        if not self._preflight_passed:
            QMessageBox.warning(
                self, 'Preflight Required',
                'Run Preflight Check and resolve issues before deploying.')
            return
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
        if not self._preflight_passed:
            QMessageBox.warning(
                self, 'Preflight Required',
                'Run Preflight Check and resolve issues before training.')
            return
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

    def _image_path(self, image_name):
        return str(self._GUI_DIR / 'img' / image_name)
