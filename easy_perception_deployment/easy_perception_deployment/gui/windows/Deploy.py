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

import os
import sys
import json
import subprocess
import logging
import threading
import time
from collections import deque

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from epd_msgs.msg import EPDObjectDetection, EPDObjectLocalization, EPDObjectTracking
    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False
from PySide6.QtCore import QObject, QSize, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QLabel,
                               QMessageBox, QPushButton, QWidget,
                               QDoubleSpinBox, QSpinBox)

from windows.Counting import CountingWindow
from windows.Tracking import TrackingWindow


class _FPSMonitorSignals(QObject):
    """QObject carrier for the fps_updated signal.

    Kept separate from the worker thread so the signal lives on the main thread
    and cross-thread emissions are automatically queued by Qt — regardless of
    whether the emitting thread is a QThread or a plain Python thread.
    """

    fps_updated = Signal(str)


class FPSMonitorThread:
    """FPS monitor that runs in a daemon Python thread (not a QThread).

    Using threading.Thread avoids the QThread destructor calling abort() when
    the C++ thread object is destroyed while the OS thread is still running —
    a crash that occurs in PySide6 6.x when the thread is doing heavy ROS2
    node initialisation and the owning widget is closed before init completes.
    """

    def __init__(self, usecase_mode):
        self._signals = _FPSMonitorSignals()
        self.fps_updated = self._signals.fps_updated

        self._usecase_mode = usecase_mode
        self._requested_mode = usecase_mode
        self._node = None
        self._subscription = None
        self._stamps = deque(maxlen=30)
        self._running = True
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name='FPSMonitor')

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    def wait(self, timeout_ms=None):
        """Block until the thread finishes; return True if it stopped in time."""
        timeout_sec = timeout_ms / 1000.0 if timeout_ms is not None else None
        self._thread.join(timeout=timeout_sec)
        return not self._thread.is_alive()

    def set_usecase_mode(self, usecase_mode):
        with self._lock:
            self._requested_mode = usecase_mode

    def _run(self):
        if not _RCLPY_AVAILABLE:
            return
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = Node('epd_fps_monitor')
            self._update_subscription(self._usecase_mode)
            while rclpy.ok() and self._running:
                self._maybe_update_subscription()
                rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception as exc:
            logging.getLogger('deploy').warning(
                'FPS monitor thread failed: %s', exc)
            self.fps_updated.emit('FPS: N/A | Latency: N/A (ROS error)')
        finally:
            if self._subscription is not None:
                try:
                    self._node.destroy_subscription(self._subscription)
                except Exception as e:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error destroying subscription: %s', e)
                self._subscription = None
            if self._node is not None:
                try:
                    self._node.destroy_node()
                except Exception as e:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error destroying node: %s', e)
                self._node = None
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logging.getLogger('deploy').debug(
                    'FPS monitor: error during rclpy shutdown: %s', e)

    def _maybe_update_subscription(self):
        with self._lock:
            requested = self._requested_mode
        if requested != self._usecase_mode:
            self._usecase_mode = requested
            self._update_subscription(self._usecase_mode)

    def _update_subscription(self, usecase_mode):
        if self._node is None:
            return
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
        topic, msg_type = self._topic_for_usecase(usecase_mode)
        self._stamps.clear()
        if topic is None:
            self.fps_updated.emit('FPS: -- | Latency: --')
            return
        self._subscription = self._node.create_subscription(
            msg_type,
            topic,
            self._message_callback,
            10)

    def _topic_for_usecase(self, usecase_mode):
        if usecase_mode in (0, 1):
            return '/easy_perception_deployment/p2_inference', EPDObjectDetection
        if usecase_mode == 2:
            return '/easy_perception_deployment/p3_inference', EPDObjectDetection
        if usecase_mode == 3:
            return '/easy_perception_deployment/localization', EPDObjectLocalization
        if usecase_mode == 4:
            return '/easy_perception_deployment/tracking', EPDObjectTracking
        return None, None

    def _message_callback(self, msg):
        timestamp = self._timestamp_from_msg(msg)
        if timestamp is not None:
            self._stamps.append(timestamp)
        fps = self._compute_fps()
        latency_ms = self._process_time_ms(msg)
        fps_text = f'{fps:.1f}' if fps is not None else '--'
        latency_text = f'{latency_ms:.1f} ms' if latency_ms is not None else '--'
        self.fps_updated.emit(f'FPS: {fps_text} | Latency: {latency_text}')

    def _timestamp_from_msg(self, msg):
        if not hasattr(msg, 'header'):
            return None
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return time.time()
        return Time.from_msg(stamp).nanoseconds / 1e9

    def _compute_fps(self):
        if len(self._stamps) < 2:
            return None
        delta = self._stamps[-1] - self._stamps[0]
        if delta <= 0:
            return None
        return (len(self._stamps) - 1) / delta

    def _process_time_ms(self, msg):
        if hasattr(msg, 'process_time') and msg.process_time:
            return float(msg.process_time)
        return None


class DeployWindow(QWidget):
    '''
    The DeployWindow class is a PySide6 Graphical User Interface (GUI) window
    that is called by MainWindow class in order to configure a custom session
    and write to session_config.json.
    '''
    def __init__(self, debug=False):
        '''
        The constructor.
        Sets the size of the window and configurations for session_config.
        Checks if the session_config.json file exists. If true, configure
        accordingly. Otherwise, assign default values.
        Calls setButtons function to populate window with button.
        '''
        super().__init__()

        self.deploy_logger = logging.getLogger('deploy')

        self.debug = debug

        self._path_to_model = ''
        self._path_to_label_list = ''
        self._input_image_topic = ''
        self._image_transport = 'raw'

        self._is_running = False

        self._DEPLOY_WIN_H = 540
        self._DEPLOY_WIN_W = 500

        self.setWindowIcon(QIcon("img/epd_desktop.png"))

        self._deploy_process = None
        self._kill_process = None
        self._deploy_timer = None
        self._kill_timer = None
        self._fps_monitor = None

        self.visualizeFlag = True

        self.useCPU = True
        self._intra_op_num_threads = 0
        self.publish_detection_segmentation = True
        self._confidence_threshold = 0.5
        self._max_detections = 100

        self._path_to_session_config = ('../config/session_config.json')
        self._path_to_usecase_config = ('../config/usecase_config.json')
        self._path_to_input_image_json_file = (
            '../config/input_image_topic.json')

        self.usecase_list = [
            'Classification',
            'Counting',
            'Color-Matching',
            'Localization',
            'Tracking']
        self.image_transport_list = [
            'raw',
            'compressed']

        session_config = None
        usecase_config = None
        if self.doesFileExist(self._path_to_session_config):
            session_config_json_obj = open(self._path_to_session_config)
            session_config = json.load(session_config_json_obj)
        else:
            self.deploy_logger.warning(
                '[ session_config.json ] is missing.' +
                'Assigning default values')
            self._path_to_model = 'filepath/to/onnx/model'
            self._path_to_label_list = 'filepath/to/classes/list/txt'
            self.visualizeFlag = True
            self.useCPU = True

        if self.doesFileExist(self._path_to_usecase_config):
            usecase_config_json_obj = open(self._path_to_usecase_config)
            usecase_config = json.load(usecase_config_json_obj)
        else:
            self.deploy_logger.warning(
                '[usecase_config.json] is missing.'
                'Assigning default Use Case MODE : ' +
                '[CLASSIFICATION] ')
            self.usecase_mode = 0

        try:
            self._path_to_model = session_config["path_to_model"]
            self._path_to_label_list = session_config["path_to_label_list"]
            self._intra_op_num_threads = session_config.get(
                "intra_op_num_threads",
                0)
            self._image_transport = session_config.get(
                "image_transport",
                "raw").lower()
            self.publish_detection_segmentation = session_config.get(
                "publish_detection_segmentation",
                True)
            self._confidence_threshold = float(session_config.get(
                "confidence_threshold", 0.5))
            self._max_detections = int(session_config.get(
                "max_detections", 100))
            if session_config["visualizeFlag"] == "visualize":
                self.visualizeFlag = True
            else:
                self.visualizeFlag = False
            if session_config["useCPU"] == "CPU":
                self.useCPU = True
            else:
                self.useCPU = False
        except (KeyError, TypeError) as e:
            self.deploy_logger.exception("[ session_config.json ] - " +
                                         "KeyError or TypeError detected" +
                                         "Assigning default values")
            self._path_to_model = 'filepath/to/onnx/model'
            self._path_to_label_list = 'filepath/to/classes/list/txt'
            self.visualizeFlag = True
            self.useCPU = True
            self._intra_op_num_threads = 0
            self._image_transport = 'raw'
            self._confidence_threshold = 0.5
            self._max_detections = 100

        try:
            self.usecase_mode = int(usecase_config["usecase_mode"])

            if self.usecase_mode < 0 or self.usecase_mode > 4:
                self.deploy_logger.warning(
                    '[ usecase_config.json ] - Invalid Usecase Mode' +
                    ' - FOUND.\n'
                    'Assigning default Use Case MODE : [CLASSIFICATION] ')
                self.usecase_mode = 0

            # Rearranging usecase_list based on saved configuration.
            curr_usecase_mode = self.usecase_list[int(self.usecase_mode)]
            self.usecase_list.remove(curr_usecase_mode)
            self.usecase_list.insert(0, curr_usecase_mode)
        except TypeError:
            self.deploy_logger.exception(
                "[ usecase_config.json ] - " +
                "TypeError detected" +
                "Assigning default Use Case MODE : " +
                "[CLASSIFICATION] ")
            self.usecase_mode = 0

        if self.doesFileExist(self._path_to_input_image_json_file):
            # Load input_image_topic.json
            f = open(self._path_to_input_image_json_file)
            data = json.load(f)
            self._input_image_topic = data['input_image_topic']
        else:
            self._input_image_topic = '/camera/color/image_raw'

        if self._image_transport not in self.image_transport_list:
            self.deploy_logger.warning(
                '[ session_config.json ] - Invalid image_transport. '
                'Defaulting to raw.')
            self._image_transport = 'raw'

        if self._image_transport in self.image_transport_list:
            self.image_transport_list.remove(self._image_transport)
        self.image_transport_list.insert(0, self._image_transport)

        self.setWindowTitle('Deploy')
        self.setGeometry(self._DEPLOY_WIN_W,
                         0,
                         self._DEPLOY_WIN_W,
                         self._DEPLOY_WIN_H)
        self.setFixedSize(self._DEPLOY_WIN_W, self._DEPLOY_WIN_H)

        self.setButtons()
        self._start_fps_monitor()
        self.validateDeployInputs()
        self.printDeployConfig()

    def printDeployConfig(self):
        '''
        A Non-Return Getter function that prints EPD Deployment
        configurations that are not displayed clearly in EPD GUI, on terminal.
        '''
        self.deploy_logger.info('[- EPD Deployment Configurations -]')
        self.deploy_logger.info('[ ONNX Model ] : ' + self._path_to_model)
        self.deploy_logger.info('[ Label List ] : ' + self._path_to_label_list)
        self.deploy_logger.info(
            '[ Input Image Topic ] : ' + self._input_image_topic)
        self.deploy_logger.info(
            '[ Image Transport ] : ' + self._image_transport)
        self.deploy_logger.info(
            '[ Detection Segmentation ] : ' +
            ('Enabled' if self.publish_detection_segmentation else 'Disabled'))

    def setButtons(self):
        '''A Mutator function that defines all buttons in DeployWindow.'''
        # ONNX Model to set the path to ONNX model and
        # store in session_config.json
        self.model_button = QPushButton('ONNX Model', self)
        self.model_button.setIcon(QIcon('img/model.png'))
        self.model_button.setIconSize(QSize(75, 75))
        self.model_button.setMinimumHeight(80)

        index = self._path_to_model.find('data/model')
        if index == -1:
            model_path = self._path_to_model
        else:
            model_path = '../' + self._path_to_model[index:]
        if self.doesFileExist(model_path):
            self.model_button.setStyleSheet(
                'background-color: rgba(0,150,10,255);')
        else:
            self.model_button.setStyleSheet(
                'background-color: rgba(200,10,0,255);')

        # Label List to set the path to ONNX model
        # and store in session_config.json
        self.list_button = QPushButton('Label List', self)
        self.list_button.setIcon(QIcon('img/label_list.png'))
        self.list_button.setIconSize(QSize(75, 75))
        self.list_button.setMinimumHeight(80)

        index = self._path_to_label_list.find('data/label_list')
        if index == -1:
            label_list_path = self._path_to_label_list
        else:
            label_list_path = '../' + self._path_to_label_list[index:]
        if self.doesFileExist(label_list_path):
            self.list_button.setStyleSheet(
                'background-color: rgba(0,150,10,255);')
        else:
            self.list_button.setStyleSheet(
                'background-color: rgba(200,10,0,255);')

        # UseCase Config Dropdown to select usecase mode
        self.usecase_config_button = QComboBox(self)
        for usecase in self.usecase_list:
            self.usecase_config_button.addItem(usecase)

        if self.doesFileExist(self._path_to_usecase_config):
            self.usecase_config_button.setStyleSheet(
                'background-color: rgba(0,150,10,255);')
        else:
            self.usecase_config_button.setStyleSheet(
                'background-color: rgba(200,10,0,255);')

        self.visualize_button = QPushButton(self)
        self.visualize_button.setMinimumHeight(80)
        if self.visualizeFlag:
            self.visualize_button.setText('Visualize')
        else:
            self.visualize_button.setText('Action')

        self.segmentation_button = QPushButton(self)
        self.segmentation_button.setMinimumHeight(40)
        if self.publish_detection_segmentation:
            self.segmentation_button.setText('Segmentation On')
        else:
            self.segmentation_button.setText('Segmentation Off')

        self.refresh_topics_button = QPushButton(self)
        self.refresh_topics_button.setText('Refresh topics')

        self.topic_button = QComboBox(self)
        self.topic_button.setEditable(True)
        self.topic_button.setInsertPolicy(QComboBox.NoInsert)
        self.topic_button.setFixedHeight(28)
#         self.refreshImageTopics(select_topic=self._input_image_topic)

        self.transport_label = QLabel('Image Transport', self)
        self.transport_combo = QComboBox(self)
        for transport in self.image_transport_list:
            self.transport_combo.addItem(transport)

        self.docker_button = QPushButton(self)
        if self.useCPU is True:
            self.docker_button.setText('CPU')
        else:
            self.docker_button.setText('GPU')

        # Confidence threshold spinbox
        self.confidence_label = QLabel('Confidence Threshold', self)
        self.confidence_spinbox = QDoubleSpinBox(self)
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self._confidence_threshold)

        # Max detections spinbox
        self.max_detections_label = QLabel('Max Detections (0 = unlimited)', self)
        self.max_detections_spinbox = QSpinBox(self)
        self.max_detections_spinbox.setRange(0, 10000)
        self.max_detections_spinbox.setSingleStep(10)
        self.max_detections_spinbox.setValue(self._max_detections)

        # Validation label - shows validation messages
        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)

        # Status label - shows run status (Stopped/Running)
        self.status_label = QLabel('Stopped', self)
        self.status_label.setIndent(10)

        # FPS/Latency label
        self.fps_label = QLabel('FPS: -- | Latency: --', self)
        self.fps_label.setIndent(10)

        # Run button to deploy ROS2 package with info
        # from usecase_config.json and session_config.json
        self.run_button = QPushButton('Run', self)
        self.run_button.setIcon(QIcon('img/go.png'))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.setMinimumHeight(100)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.model_button, 0, 0)
        layout.addWidget(self.list_button, 0, 1)
        layout.addWidget(self.visualize_button, 1, 0)
        layout.addWidget(self.usecase_config_button, 1, 1)
        layout.addWidget(self.segmentation_button, 2, 0)
        layout.addWidget(self.refresh_topics_button, 2, 1)
        layout.addWidget(self.topic_button, 3, 0, 1, 2)
        layout.addWidget(self.transport_label, 4, 0)
        layout.addWidget(self.transport_combo, 4, 1)
        layout.addWidget(self.docker_button, 5, 0, 1, 2)
        layout.addWidget(self.confidence_label, 6, 0)
        layout.addWidget(self.confidence_spinbox, 6, 1)
        layout.addWidget(self.max_detections_label, 7, 0)
        layout.addWidget(self.max_detections_spinbox, 7, 1)
        layout.addWidget(self.validation_label, 8, 0, 1, 2)
        layout.addWidget(self.status_label, 9, 0, 1, 2)
        layout.addWidget(self.fps_label, 10, 0, 1, 2)
        layout.addWidget(self.run_button, 11, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        # Connect signals to slots
        self.visualize_button.clicked.connect(self.setVisualizeFlag)
        self.segmentation_button.clicked.connect(self.toggleSegmentationPublishing)
        self.docker_button.clicked.connect(self.setDockerFlag)
        self.model_button.clicked.connect(self.setModel)
        self.list_button.clicked.connect(self.setLabelList)
        self.usecase_config_button.activated.connect(self.setUseCase)
        self.run_button.clicked.connect(self.deployPackage)
#         self.register_topic_button.clicked.connect(self.setImageInput)
        self.transport_combo.activated.connect(self.setImageTransport)
        self.refresh_topics_button.clicked.connect(self.refreshImageTopics)
        self.topic_button.currentTextChanged.connect(self.setImageInput)
        self.confidence_spinbox.valueChanged.connect(self._onConfidenceChanged)
        self.max_detections_spinbox.valueChanged.connect(self._onMaxDetectionsChanged)

        # Populate topics after widgets are created (avoids run_button init order issues)
        self.refreshImageTopics(select_topic=self._input_image_topic)

    def _query_image_topics(self):
        topics = []
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list', '-t'],
                capture_output=True,
                text=True,
                check=False)
        except FileNotFoundError:
            self.deploy_logger.warning('ros2 command not found.')
            return topics

        if result.returncode != 0:
            self.deploy_logger.warning(
                'Failed to query ROS2 topics: %s', result.stderr.strip())
            return topics

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if 'sensor_msgs/msg/Image' not in stripped:
                continue
            topic_name = stripped.split()[0]
            topics.append(topic_name)
        return topics

    def refreshImageTopics(self, select_topic=None):
        topics = self._query_image_topics()
        current_topic = (
            select_topic
            if select_topic is not None
            else self.topic_button.currentText().strip())

        self.topic_button.blockSignals(True)
        self.topic_button.clear()

        if topics:
            self.topic_button.addItems(topics)
        elif current_topic:
            self.topic_button.addItem(current_topic)

        if current_topic:
            index = self.topic_button.findText(current_topic)
            if index != -1:
                self.topic_button.setCurrentIndex(index)
            else:
                self.topic_button.setEditText(current_topic)
        elif self.topic_button.count() > 0:
            self.topic_button.setCurrentIndex(0)

        self.topic_button.blockSignals(False)
        self.setImageInput()

    def _start_fps_monitor(self):
        if not _RCLPY_AVAILABLE:
            self.fps_label.setText('FPS: N/A | Latency: N/A (ROS unavailable)')
            return
        self._fps_monitor = FPSMonitorThread(self.usecase_mode)
        self._fps_monitor.fps_updated.connect(self._update_fps_label)
        self._fps_monitor.start()

    @Slot(str)
    def _update_fps_label(self, text):
        self.fps_label.setText(text)

    def _update_fps_monitor_mode(self, usecase_mode):
        if self._fps_monitor is not None:
            self._fps_monitor.set_usecase_mode(usecase_mode)

    def _stop_fps_monitor(self):
        if self._fps_monitor is None:
            return
        self._fps_monitor.stop()
        self._fps_monitor.wait(1000)
        self._fps_monitor = None

    def deployPackage(self):
        '''
        A Mutator function that runs a bash script that
        checks if the _is_running boolean flag is True or not.\n
        If False, run bash script to run ROS2 package with
        session_config.json and usecase_config.json
        Otherwise, run bash script to kill ROS2 package
        processes remotely.
        '''
        if not self._is_running:
            self._deploy_process, self._deploy_timer = self._start_process(
                [self._deploy_script_path(),
                 str(self.useCPU),
                 str(self.visualizeFlag)],
                'deploy',
                cwd=self._scripts_dir())
            self.run_button.setText('Stop')
            self.run_button.setIcon(QIcon('img/quit.png'))
            self.run_button.setIconSize(QSize(100, 100))
            self.run_button.updateGeometry()
            self._is_running = True
            self.status_label.setText('Running...')
        else:
            self._stop_deployment()

    def _stop_deployment(self):
        self.deploy_logger.info("Killing epd_test_container docker.")
        self._kill_process, self._kill_timer = self._start_process(
            [self._kill_script_path()],
            'kill',
            cwd=self._scripts_dir())
        self.run_button.setText('Run')
        self.run_button.setIcon(QIcon('img/go.png'))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.updateGeometry()
        self._is_running = False
        self.status_label.setText('Stopped')

    def _scripts_dir(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts"))

    def _deploy_script_path(self):
        return os.path.join(self._scripts_dir(), "deploy.sh")

    def _kill_script_path(self):
        return os.path.join(self._scripts_dir(), "kill.sh")

    def _start_process(self, args, process_type, cwd=None):
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd)
        timer = QTimer(self)
        timer.setInterval(200)
        timer.timeout.connect(
            lambda: self._check_process(process, timer, process_type))
        timer.start()
        return process, timer

    def _check_process(self, process, timer, process_type):
        if process.poll() is None:
            return

        timer.stop()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self._handle_process_error(process_type, stdout, stderr)
            return

        if process_type == 'kill':
            self.status_label.setText('Stopped')
        elif process_type == 'deploy':
            self.status_label.setText('Running...')

    def _handle_process_error(self, process_type, stdout, stderr):
        self.run_button.setText('Run')
        self.run_button.setIcon(QIcon('img/go.png'))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.updateGeometry()
        self._is_running = False
        self.status_label.setText('Error')

        stdout_text = stdout.strip()
        stderr_text = stderr.strip()
        message_lines = [
            f"{process_type.capitalize()} failed.",
            "",
            "stdout:",
            stdout_text if stdout_text else "(empty)",
            "",
            "stderr:",
            stderr_text if stderr_text else "(empty)"
        ]
        msgBox = QMessageBox()
        msgBox.setText('\n'.join(message_lines))
        msgBox.exec()

    def _terminate_process(self, process, timer):
        if timer is not None and timer.isActive():
            timer.stop()
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def shutdown(self):
        if self._is_running:
            self._stop_deployment()

        if self._kill_process is not None and self._kill_process.poll() is None:
            try:
                self._kill_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._terminate_process(self._kill_process, self._kill_timer)
        else:
            self._terminate_process(self._kill_process, self._kill_timer)

        self._terminate_process(self._deploy_process, self._deploy_timer)
        self._deploy_process = None
        self._kill_process = None
        self._deploy_timer = None
        self._kill_timer = None

    def closeEvent(self, event):
        self.shutdown()
        event.accept()

    def setImageInput(self):
        '''
        A Mutator function that writes to line 25 of
        run.launch.py file based on new image topic.
        '''
        new_image_topic = self.topic_button.currentText().strip()
        if new_image_topic and self.topic_button.findText(new_image_topic) == -1:
            self.topic_button.addItem(new_image_topic)
        self.deploy_logger.info(
            'Rewriting Input Image Topic to: ' +
            new_image_topic)

        if not new_image_topic:
            self._input_image_topic = ''
            self.validateDeployInputs()
            return

        dict = {"input_image_topic": new_image_topic}
        json_object = json.dumps(dict, indent=4)
        self._write_json_atomic(self._path_to_input_image_json_file, json_object)

        self._input_image_topic = new_image_topic
        self.validateDeployInputs()

    def doesFileExist(self, input_filepath):
        ''' A Getter function that checks if a given file exists.'''
        if os.path.exists(input_filepath):
            return True
        else:
            self.deploy_logger.warning(
                '[ ' + input_filepath + ' ] does not exist.')
            return False

    def setVisualizeFlag(self):
        '''A function is triggered by the button labelled, Visualize/Action.'''
        self.visualizeFlag = not self.visualizeFlag

        if self.visualizeFlag:
            self.visualize_button.setText('Visualize')
        else:
            self.visualize_button.setText('Action')

        self.visualize_button.updateGeometry()
        self.updateSessionConfig()

    def toggleSegmentationPublishing(self):
        '''A function is triggered by the button labelled, Segmentation.'''
        self.publish_detection_segmentation = not self.publish_detection_segmentation

        if self.publish_detection_segmentation:
            self.segmentation_button.setText('Segmentation On')
        else:
            self.segmentation_button.setText('Segmentation Off')

        self.segmentation_button.updateGeometry()
        self.updateSessionConfig()

    def setDockerFlag(self):
        '''A function is triggered by the button labelled, CPU/GPU.'''
        self.useCPU = not self.useCPU

        if self.useCPU:
            self.docker_button.setText('CPU')
        else:
            self.docker_button.setText('GPU')
        self.docker_button.updateGeometry()
        self.updateSessionConfig()

    def setUseCase(self, index):
        '''A function is triggered by the DropDown Menu labelled, UseCase.'''
        selected_usecase = self.usecase_list[index]

        if selected_usecase == 'Classification':
            self.usecase_mode = 0
            if not self.debug:
                msgBox = QMessageBox()
                msgBox.setText('[Classification] Selected.'
                               'No other configuration required.')
                msgBox.exec()

            self.deploy_logger.info('Wrote to ../data/usecase_config.json')
            dict = {"usecase_mode": 0}
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)

        elif selected_usecase == 'Counting':
            self.usecase_mode = 1
            self.counting_window = CountingWindow(self._path_to_label_list,
                                                  self._path_to_usecase_config)
            self.counting_window.show()
        elif selected_usecase == 'Localization':
            self.usecase_mode = 3
            dict = {"usecase_mode": 3}
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)
        elif selected_usecase == 'Tracking':
            self.usecase_mode = 4
            self.tracking_window = TrackingWindow(self._path_to_usecase_config)
            self.tracking_window.show()
        elif selected_usecase == 'Color-Matching':
            self.usecase_mode = 2
            if not self.debug:
                input_refimage_filepath, ok = (
                    QFileDialog.getOpenFileName(
                        self,
                        'Set the .png/.jpg color image to use',
                        os.path.abspath('../data'),
                        'Image Files (*.png *.jpg *.jpeg)'))
            else:
                input_refimage_filepath = 'dummy_filepath_to_refimage'
                ok = True

            if ok:
                filepath_index = input_refimage_filepath.find('/data')
                path_to_color_template = (
                    '.' +
                    input_refimage_filepath[filepath_index:])
            else:
                self.deploy_logger.warning('No reference color template set.')
                return

            self.deploy_logger.info('Wrote to ../data/usecase_config.json')
            dict = {
                "usecase_mode": 2,
                "path_to_color_template": path_to_color_template,
                "color_match_histogram_metric": "Correlation"
                }
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)
        else:
            self.deploy_logger.warning('Invalid Use Case')
            sys.exit()

        self.usecase_config_button.setStyleSheet(
            'background-color: rgba(0,150,10,255);')
        self._update_fps_monitor_mode(self.usecase_mode)

    def setImageTransport(self, index):
        '''A function triggered by the Image Transport dropdown.'''
        self._image_transport = self.image_transport_list[index]
        self.updateSessionConfig()

    def _onConfidenceChanged(self, value):
        '''Slot triggered when the confidence threshold spinbox changes.'''
        self._confidence_threshold = float(value)
        self.updateSessionConfig()

    def _onMaxDetectionsChanged(self, value):
        '''Slot triggered when the max detections spinbox changes.'''
        self._max_detections = int(value)
        self.updateSessionConfig()

    def _write_json_atomic(self, path, json_str):
        '''Write json_str to path atomically using a .tmp file + os.replace().'''
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w') as outfile:
            outfile.write(json_str)
        os.replace(tmp_path, path)

    def updateSessionConfig(self):
        '''A Mutator function that updates the session_config.json file.'''

        if self.visualizeFlag:
            visualizeFlag_string = "visualize"
        else:
            visualizeFlag_string = "robot"

        if self.useCPU:
            useCPU_string = "CPU"
        else:
            useCPU_string = "GPU"

        dict = {
            "path_to_model": self._path_to_model,
            "path_to_label_list": self._path_to_label_list,
            "visualizeFlag": visualizeFlag_string,
            "useCPU": useCPU_string,
            "intra_op_num_threads": self._intra_op_num_threads,
            "image_transport": self._image_transport,
            "publish_detection_segmentation": (
                self.publish_detection_segmentation),
            "confidence_threshold": self._confidence_threshold,
            "max_detections": self._max_detections
            }
        json_object = json.dumps(dict, indent=4)
        self._write_json_atomic(self._path_to_session_config, json_object)

    def setModel(self):
        '''A function is triggered by the button labelled, ONNX Model.'''
        if not self.debug:
            input_model_filepath, ok = (
                QFileDialog.getOpenFileName(self,
                                            'Set the .onnx model to use',
                                            os.path.abspath('../data'),
                                            'ONNX Model Files (*.onnx)'))
        else:
            input_model_filepath = 'dummy_model_filepath'
            ok = True

        if ok:
            self._path_to_model = input_model_filepath

            index = input_model_filepath.find('/data/model')
            if index == -1:
                self._path_to_model = input_model_filepath
            else:
                self._path_to_model = '.' + input_model_filepath[index:]
        else:
            self.deploy_logger.warning('No ONNX model set.')
            return

        self.model_button.setStyleSheet(
            'background-color: rgba(0,150,10,255);')
        self.updateSessionConfig()
        self.validateDeployInputs()

    def setLabelList(self):
        '''A function is triggered by the button labelled, Label List.'''
        if not self.debug:
            input_classes_filepath, ok = (
                QFileDialog.getOpenFileName(self,
                                            'Set the .json to use',
                                            os.path.abspath('../data'),
                                            'Text Files (*.txt)'))
        else:
            input_classes_filepath = 'dummy_label_list_filepath'
            ok = True

        if not ok:
            self.deploy_logger.warning('No label list set.')
            return

        # Validate the selected label list file before accepting it.
        validation_error = self._validate_label_list_file(
            input_classes_filepath)
        if validation_error is not None:
            self.deploy_logger.error(
                'Invalid label list file [%s]: %s',
                input_classes_filepath,
                validation_error)
            if not self.debug:
                msgBox = QMessageBox()
                msgBox.setWindowTitle('Invalid Label List')
                msgBox.setText(
                    'The selected label list file is invalid:\n\n' +
                    validation_error +
                    '\n\nPlease select a valid UTF-8 text file with at '
                    'least one non-empty class label per line.')
                msgBox.exec()
            return

        index = input_classes_filepath.find('/data/label_list')
        if index == -1:
            self._path_to_label_list = input_classes_filepath
        else:
            self._path_to_label_list = '.' + input_classes_filepath[index:]

        self.list_button.setStyleSheet('background-color: rgba(0,150,10,255);')
        self.updateSessionConfig()
        self.validateDeployInputs()

    def _validate_label_list_file(self, filepath):
        '''
        Validate a label list text file.
        Returns None if valid, or an error string describing the problem.
        '''
        if not filepath or filepath == 'dummy_label_list_filepath':
            return None
        try:
            with open(filepath, encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return 'File not found: ' + filepath
        except PermissionError:
            return 'Permission denied reading file: ' + filepath
        except UnicodeDecodeError as e:
            return 'File is not valid UTF-8 text: ' + str(e)

        non_empty = [ln.strip() for ln in lines if ln.strip()]
        if not non_empty:
            return 'File contains no non-empty class labels.'
        return None

    def resolveFilePath(self, input_filepath):
        '''Resolve a file path for validation.'''
        if not input_filepath:
            return ''
        expanded_path = os.path.expandvars(os.path.expanduser(input_filepath))
        return os.path.abspath(expanded_path)

    def validateDeployInputs(self):
        '''Validate inputs and update the Run button state.'''
        if self._is_running:
            self.run_button.setEnabled(True)
            self.run_button.setToolTip('')
            self.validation_label.setText('')
            return

        missing_items = []

        model_path = self.resolveFilePath(self._path_to_model)
        if not model_path or not os.path.isfile(model_path):
            missing_items.append('ONNX model file')

        label_list_path = self.resolveFilePath(self._path_to_label_list)
        if not label_list_path or not os.path.isfile(label_list_path):
            missing_items.append('label list file')

        if not self._input_image_topic.strip():
            missing_items.append('input image topic')

        if missing_items:
            message = 'Missing: ' + ', '.join(missing_items)
            self.run_button.setEnabled(False)
            self.run_button.setToolTip(message)
            self.validation_label.setText(message)
        else:
            self.run_button.setEnabled(True)
            self.run_button.setToolTip('')
            self.validation_label.setText('')

    def closeEvent(self, event):
        self._stop_fps_monitor()
        super().closeEvent(event)
