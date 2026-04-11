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
from pathlib import Path
from collections import deque

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from epd_msgs.msg import EPDObjectDetection, EPDObjectLocalization, EPDObjectTracking
    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False
from PySide6.QtCore import QObject, QSize, QThread, QTimer, Signal, Slot
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


class _ImageTopicsWorkerSignals(QObject):
    """Signal carrier for async ROS topic discovery results."""

    success = Signal(list)
    error = Signal(str)
    finished = Signal()


class ImageTopicsWorker(QObject):
    """Runs `ros2 topic list -t` in a Python thread managed by QThread."""

    def __init__(self, timeout_sec=3):
        super().__init__()
        self.timeout_sec = timeout_sec
        self.signals = _ImageTopicsWorkerSignals()

    def run(self):
        topics = []
        try:
            result = subprocess.run(
                ['ros2', 'topic', 'list', '-t'],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec)
        except FileNotFoundError:
            self.signals.error.emit('Unable to refresh topics: ros2 command not found.')
            self.signals.finished.emit()
            return
        except subprocess.TimeoutExpired:
            self.signals.error.emit('Topic refresh timed out. Keeping current topic list.')
            self.signals.finished.emit()
            return
        except Exception as exc:
            self.signals.error.emit(f'Unable to refresh topics: {exc}')
            self.signals.finished.emit()
            return

        if result.returncode != 0:
            stderr = result.stderr.strip()
            detail = f' ({stderr})' if stderr else ''
            self.signals.error.emit(
                f'Unable to refresh topics from ROS2{detail}. Keeping current topic list.')
            self.signals.finished.emit()
            return

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped or 'sensor_msgs/msg/Image' not in stripped:
                continue
            topic_name = stripped.split()[0]
            topics.append(topic_name)

        self.signals.success.emit(topics)
        self.signals.finished.emit()


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
            return '/easy_perception_deployment/epd_p2_output', EPDObjectDetection
        if usecase_mode == 2:
            return '/easy_perception_deployment/epd_p3_output', EPDObjectDetection
        if usecase_mode == 3:
            return '/easy_perception_deployment/epd_localize_output', EPDObjectLocalization
        if usecase_mode == 4:
            return '/easy_perception_deployment/epd_tracking_output', EPDObjectTracking
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
    _MODULE_DIR = Path(__file__).resolve().parent
    _GUI_DIR = _MODULE_DIR.parent
    _PACKAGE_ROOT = _GUI_DIR.parent

    DEFAULT_MODEL_PATH = './data/model/MaskRCNN-10.onnx'
    DEFAULT_LABEL_LIST_PATH = './data/label_list/coco_classes.txt'
    DEFAULT_INPUT_TOPIC = '/camera/camera/color/image_raw'

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

        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))

        self._deploy_process = None
        self._kill_process = None
        self._deploy_timer = None
        self._kill_timer = None
        self._deploy_log_file = None
        self._kill_log_file = None
        self._fps_monitor = None
        self._is_shutting_down = False

        self.visualizeFlag = True

        self.useCPU = True
        self._intra_op_num_threads = 0
        self.publish_detection_segmentation = True
        self._confidence_threshold = 0.5
        self._max_detections = 100
        self._image_topics_cache = []
        self._image_topics_cache_ts = 0.0
        self._image_topics_cache_ttl_sec = 3.0
        self._topics_worker_thread = None
        self._topics_worker = None

        self._path_to_session_config = str(self._config_path('session_config.json'))
        self._path_to_usecase_config = str(self._config_path('usecase_config.json'))
        self._path_to_input_image_json_file = str(
            self._config_path('input_image_topic.json'))

        self.usecase_list = [
            'Classification',
            'Counting',
            'Color-Matching',
            'Localization',
            'Tracking']
        self.image_transport_list = [
            'raw',
            'compressed']

        session_config = self._load_json_config(
            self._path_to_session_config,
            config_name='session_config.json',
            required_keys=[
                'path_to_model',
                'path_to_label_list',
                'visualizeFlag',
                'useCPU'],
            defaults={
                'path_to_model': 'filepath/to/onnx/model',
                'path_to_label_list': 'filepath/to/classes/list/txt',
                'visualizeFlag': 'visualize',
                'useCPU': 'CPU',
                'intra_op_num_threads': 0,
                'image_transport': 'raw',
                'publish_detection_segmentation': True,
                'confidence_threshold': 0.5,
                'max_detections': 100},
            allow_missing_defaults=True,
            abort_on_json_error=True)
        usecase_config = self._load_json_config(
            self._path_to_usecase_config,
            config_name='usecase_config.json',
            required_keys=['usecase_mode'],
            defaults={'usecase_mode': 0},
            allow_missing_defaults=True,
            abort_on_json_error=True)

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

        image_topic_data = self._load_json_config(
            self._path_to_input_image_json_file,
            config_name='input_image_topic.json',
            required_keys=['input_image_topic'],
            defaults={'input_image_topic': self.DEFAULT_INPUT_TOPIC},
            allow_missing_defaults=True,
            abort_on_json_error=False)
        self._input_image_topic = image_topic_data['input_image_topic']

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
        self.model_button.setIcon(QIcon(self._image_path('model.png')))
        self.model_button.setIconSize(QSize(75, 75))
        self.model_button.setMinimumHeight(80)

        model_path = self.resolveFilePath(self._path_to_model)
        if self.doesFileExist(model_path):
            self.model_button.setStyleSheet(
                'background-color: rgba(0,150,10,255);')
        else:
            self.model_button.setStyleSheet(
                'background-color: rgba(200,10,0,255);')

        # Label List to set the path to ONNX model
        # and store in session_config.json
        self.list_button = QPushButton('Label List', self)
        self.list_button.setIcon(QIcon(self._image_path('label_list.png')))
        self.list_button.setIconSize(QSize(75, 75))
        self.list_button.setMinimumHeight(80)

        label_list_path = self.resolveFilePath(self._path_to_label_list)
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

        self.use_defaults_button = QPushButton(self)
        self.use_defaults_button.setText('Use defaults')

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

        # Readiness section (text + icon to avoid color-only signalling)
        self.readiness_header_label = QLabel('Readiness', self)
        self.model_readiness_label = QLabel(self)
        self.label_list_readiness_label = QLabel(self)
        self.topic_readiness_label = QLabel(self)
        self.model_readiness_label.setWordWrap(True)
        self.label_list_readiness_label.setWordWrap(True)
        self.topic_readiness_label.setWordWrap(True)

        # Status label - shows run status (Stopped/Running)
        self.status_label = QLabel('Stopped', self)
        self.status_label.setIndent(10)

        # FPS/Latency label
        self.fps_label = QLabel('FPS: -- | Latency: --', self)
        self.fps_label.setIndent(10)

        # Run button to deploy ROS2 package with info
        # from usecase_config.json and session_config.json
        self.run_button = QPushButton('Run', self)
        self.run_button.setIcon(QIcon(self._image_path('go.png')))
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
        layout.addWidget(self.use_defaults_button, 3, 0, 1, 2)
        layout.addWidget(self.topic_button, 4, 0, 1, 2)
        layout.addWidget(self.transport_label, 5, 0)
        layout.addWidget(self.transport_combo, 5, 1)
        layout.addWidget(self.docker_button, 6, 0, 1, 2)
        layout.addWidget(self.confidence_label, 7, 0)
        layout.addWidget(self.confidence_spinbox, 7, 1)
        layout.addWidget(self.max_detections_label, 8, 0)
        layout.addWidget(self.max_detections_spinbox, 8, 1)
        layout.addWidget(self.validation_label, 9, 0, 1, 2)
        layout.addWidget(self.status_label, 10, 0, 1, 2)
        layout.addWidget(self.fps_label, 11, 0, 1, 2)
        layout.addWidget(self.readiness_header_label, 12, 0, 1, 2)
        layout.addWidget(self.model_readiness_label, 13, 0, 1, 2)
        layout.addWidget(self.label_list_readiness_label, 14, 0, 1, 2)
        layout.addWidget(self.topic_readiness_label, 15, 0, 1, 2)
        layout.addWidget(self.run_button, 16, 0, 1, 2)
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
        self.use_defaults_button.clicked.connect(self.useDefaultDeployInputs)
        self.topic_button.currentTextChanged.connect(self.setImageInput)
        self.confidence_spinbox.valueChanged.connect(self._onConfidenceChanged)
        self.max_detections_spinbox.valueChanged.connect(self._onMaxDetectionsChanged)

        # Populate topics after widgets are created (avoids run_button init order issues)
        self.refreshImageTopics(select_topic=self._input_image_topic)

    def refreshImageTopics(self, select_topic=None):
        current_topic = (
            select_topic
            if select_topic is not None
            else self.topic_button.currentText().strip())
        now = time.time()
        if (now - self._image_topics_cache_ts) <= self._image_topics_cache_ttl_sec:
            self._apply_image_topics(self._image_topics_cache, current_topic)
            return

        if self._topics_worker_thread is not None and self._topics_worker_thread.isRunning():
            return

        self.refresh_topics_button.setEnabled(False)
        self.validation_label.setText('Refreshing topics...')

        self._topics_worker = ImageTopicsWorker(timeout_sec=3)
        self._topics_worker_thread = QThread(self)
        self._topics_worker.moveToThread(self._topics_worker_thread)

        self._topics_worker_thread.started.connect(self._topics_worker.run)
        self._topics_worker.signals.success.connect(
            lambda topics, selected=current_topic:
            self._on_topics_refresh_success(topics, selected))
        self._topics_worker.signals.error.connect(self._on_topics_refresh_error)
        self._topics_worker.signals.finished.connect(self._on_topics_refresh_finished)
        self._topics_worker.signals.finished.connect(self._topics_worker_thread.quit)
        self._topics_worker_thread.finished.connect(self._topics_worker.deleteLater)
        self._topics_worker_thread.finished.connect(self._topics_worker_thread.deleteLater)
        self._topics_worker_thread.finished.connect(self._clear_topics_worker_refs)
        self._topics_worker_thread.start()

    def _apply_image_topics(self, topics, current_topic):
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

    @Slot(list)
    def _on_topics_refresh_success(self, topics, current_topic):
        self._image_topics_cache = topics
        self._image_topics_cache_ts = time.time()
        self.validation_label.setText('')
        self._apply_image_topics(topics, current_topic)

    @Slot(str)
    def _on_topics_refresh_error(self, message):
        self.deploy_logger.warning(message)
        self.validation_label.setText(message)

    @Slot()
    def _on_topics_refresh_finished(self):
        self.refresh_topics_button.setEnabled(True)

    @Slot()
    def _clear_topics_worker_refs(self):
        self._topics_worker = None
        self._topics_worker_thread = None

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
            self.run_button.setIcon(QIcon(self._image_path('quit.png')))
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
        self.run_button.setIcon(QIcon(self._image_path('go.png')))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.updateGeometry()
        self._is_running = False
        self.status_label.setText('Stopped')

    def _scripts_dir(self):
        return self._GUI_DIR / "scripts"

    def _deploy_script_path(self):
        return str(self._scripts_dir() / "deploy.sh")

    def _kill_script_path(self):
        return str(self._scripts_dir() / "kill.sh")

    def _process_log_path(self, process_type):
        logs_dir = self._scripts_dir() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return str(logs_dir / f"{process_type}.log")

    def _close_process_log_file(self, process_type):
        log_attr = f"_{process_type}_log_file"
        log_file = getattr(self, log_attr, None)
        if log_file is not None and not log_file.closed:
            log_file.close()
        setattr(self, log_attr, None)

    def _tail_process_log(self, process_type, max_chars=2000):
        log_path = self._process_log_path(process_type)
        if not os.path.exists(log_path):
            return "(log file missing)"
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as log_file:
                content = log_file.read()
                return content[-max_chars:].strip() or "(empty)"
        except OSError as exc:
            return f"(unable to read log file: {exc})"

    def _start_process(self, args, process_type, cwd=None):
        self._close_process_log_file(process_type)
        log_path = self._process_log_path(process_type)
        log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
        setattr(self, f"_{process_type}_log_file", log_file)
        process = subprocess.Popen(
            args,
            stdout=log_file,
            stderr=log_file,
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
        self._close_process_log_file(process_type)
        if process_type == 'kill':
            kill_result = self._get_kill_status_from_log()
            if process.returncode == 0:
                self.status_label.setText(
                    'Stopped' if kill_result == 'STOPPED'
                    else 'Stopped (already stopped)')
                return
            if process.returncode == 2 and kill_result == 'PARTIAL_CLEANUP':
                self.status_label.setText('Stopped (partial cleanup)')
                self._show_kill_partial_cleanup_warning()
                return

        if process.returncode != 0:
            self._handle_process_error(process_type)
            return

        if process_type == 'kill':
            self.status_label.setText('Stopped')
        elif process_type == 'deploy':
            self.status_label.setText('Running...')

    def _get_kill_status_from_log(self):
        kill_log = self._tail_process_log('kill', max_chars=4000)
        if 'STATUS: PARTIAL_CLEANUP' in kill_log:
            return 'PARTIAL_CLEANUP'
        if 'STATUS: ALREADY_STOPPED' in kill_log:
            return 'ALREADY_STOPPED'
        if 'STATUS: STOPPED' in kill_log:
            return 'STOPPED'
        return None

    def _show_kill_partial_cleanup_warning(self):
        log_tail = self._tail_process_log('kill')
        message_lines = [
            "Stop completed with partial cleanup.",
            "",
            "Some processes may still be running. See log output:",
            log_tail
        ]
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText('\n'.join(message_lines))
        msgBox.exec()

    def _handle_process_error(self, process_type):
        self.run_button.setText('Run')
        self.run_button.setIcon(QIcon(self._image_path('go.png')))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.updateGeometry()
        self._is_running = False
        self.status_label.setText('Error')

        log_tail = self._tail_process_log(process_type)
        message_lines = [
            f"{process_type.capitalize()} failed.",
            "",
            "log output:",
            log_tail
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
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        try:
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
            self._close_process_log_file('deploy')
            self._close_process_log_file('kill')
            self._deploy_process = None
            self._kill_process = None
            self._deploy_timer = None
            self._kill_timer = None
        finally:
            self._is_shutting_down = False

    # Keep exactly one closeEvent override in this class; duplicate overrides
    # can silently shadow each other and skip important shutdown steps.
    def closeEvent(self, event):
        self.shutdown()
        self._stop_fps_monitor()
        super().closeEvent(event)

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
                        str(self._data_dir()),
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
                                            str(self._data_dir()),
                                            'ONNX Model Files (*.onnx)'))
        else:
            input_model_filepath = 'dummy_model_filepath'
            ok = True

        if ok:
            self._path_to_model = self._normalize_data_path(input_model_filepath)
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
                                            str(self._data_dir()),
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

        self._path_to_label_list = self._normalize_data_path(input_classes_filepath)

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
        if os.path.isabs(expanded_path):
            return expanded_path
        return str((self._PACKAGE_ROOT / expanded_path).resolve())

    def _data_dir(self):
        return self._PACKAGE_ROOT / 'data'

    def _normalize_data_path(self, filepath):
        if not filepath:
            return filepath
        resolved = Path(filepath).expanduser().resolve()
        try:
            relative_to_data = resolved.relative_to(self._data_dir())
            return str(Path('.') / 'data' / relative_to_data)
        except ValueError:
            return str(resolved)

    def _config_path(self, filename):
        return self._PACKAGE_ROOT / 'config' / filename

    def _image_path(self, image_name):
        return str(self._GUI_DIR / 'img' / image_name)

    def _set_readiness_row(self, label, name, is_ready, detail):
        state_icon = '✅' if is_ready else '❌'
        detail_text = detail if detail else '(not set)'
        label.setText(f'{name}: {state_icon} {detail_text}')

    def _set_ready_style(self, widget, is_ready):
        if is_ready:
            widget.setStyleSheet('background-color: rgba(0,150,10,255);')
        else:
            widget.setStyleSheet('background-color: rgba(200,10,0,255);')

    def _load_json_config(
            self,
            path,
            config_name,
            required_keys=None,
            defaults=None,
            allow_missing_defaults=False,
            abort_on_json_error=False):
        required_keys = required_keys or []
        defaults = defaults or {}

        if not self.doesFileExist(path):
            if allow_missing_defaults:
                self.deploy_logger.warning(
                    '[ %s ] is missing. Assigning default values.',
                    config_name)
                return dict(defaults)
            self._show_blocking_config_error(
                f'Missing required configuration file: {config_name}.',
                abort_operation=True)

        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            message = (f'{config_name} contains malformed JSON: {e}.')
            if abort_on_json_error:
                self._show_blocking_config_error(message, abort_operation=True)
            self.deploy_logger.warning(
                '[ %s ] malformed JSON. Using defaults. Details: %s',
                config_name,
                e)
            return dict(defaults)
        except OSError as e:
            if allow_missing_defaults:
                self.deploy_logger.warning(
                    '[ %s ] read error. Using defaults. Details: %s',
                    config_name,
                    e)
                return dict(defaults)
            self._show_blocking_config_error(
                f'Unable to read {config_name}: {e}.',
                abort_operation=True)

        if not isinstance(data, dict):
            if abort_on_json_error:
                self._show_blocking_config_error(
                    f'{config_name} must contain a JSON object.',
                    abort_operation=True)
            self.deploy_logger.warning(
                '[ %s ] is not a JSON object. Using defaults.',
                config_name)
            return dict(defaults)

        merged = dict(defaults)
        merged.update(data)
        missing_keys = [key for key in required_keys if key not in merged]
        if missing_keys:
            if allow_missing_defaults:
                self.deploy_logger.warning(
                    '[ %s ] missing required keys %s. Applying defaults.',
                    config_name,
                    missing_keys)
                return merged
            self._show_blocking_config_error(
                f'{config_name} missing required keys: {missing_keys}.',
                abort_operation=True)
        return merged

    def _show_blocking_config_error(self, message, abort_operation=False):
        self.deploy_logger.error(message)
        if not self.debug:
            msg_box = QMessageBox()
            msg_box.setWindowTitle('Configuration Error')
            msg_box.setText(message)
            msg_box.exec()
        if abort_operation:
            raise RuntimeError(message)

    def useDefaultDeployInputs(self):
        '''Restore default model, label list, and image topic values in one click.'''
        self._path_to_model = self.DEFAULT_MODEL_PATH
        self._path_to_label_list = self.DEFAULT_LABEL_LIST_PATH
        self._input_image_topic = self.DEFAULT_INPUT_TOPIC
        self.topic_button.setEditText(self.DEFAULT_INPUT_TOPIC)
        self.updateSessionConfig()
        dict = {"input_image_topic": self.DEFAULT_INPUT_TOPIC}
        json_object = json.dumps(dict, indent=4)
        self._write_json_atomic(self._path_to_input_image_json_file, json_object)
        self.validateDeployInputs()

    def validateDeployInputs(self):
        '''Validate inputs and update the Run button state.'''
        if self._is_running:
            self.run_button.setEnabled(True)
            self.run_button.setToolTip('')
            self.validation_label.setText('Run enabled: deployment is currently running.')
            return

        model_path = self.resolveFilePath(self._path_to_model)
        model_ok = bool(model_path) and os.path.isfile(model_path)
        label_list_path = self.resolveFilePath(self._path_to_label_list)
        label_ok = bool(label_list_path) and os.path.isfile(label_list_path)
        topic_text = self._input_image_topic.strip()
        topic_ok = bool(topic_text)

        self._set_readiness_row(
            self.model_readiness_label, 'ONNX model', model_ok, model_path)
        self._set_readiness_row(
            self.label_list_readiness_label, 'Label list', label_ok, label_list_path)
        self._set_readiness_row(
            self.topic_readiness_label, 'Input topic', topic_ok, topic_text)
        self._set_ready_style(self.model_button, model_ok)
        self._set_ready_style(self.list_button, label_ok)

        if not model_ok:
            message = f'Run disabled: model file not found at {model_path or "(not set)"}'
            self.run_button.setEnabled(False)
            self.run_button.setToolTip(message)
            self.validation_label.setText(message)
        elif not label_ok:
            message = f'Run disabled: label list file not found at {label_list_path or "(not set)"}'
            self.run_button.setEnabled(False)
            self.run_button.setToolTip(message)
            self.validation_label.setText(message)
        elif not topic_ok:
            message = 'Run disabled: input topic is empty. Set a ROS image topic.'
            self.run_button.setEnabled(False)
            self.run_button.setToolTip(message)
            self.validation_label.setText(message)
        else:
            self.run_button.setEnabled(True)
            self.run_button.setToolTip('')
            self.validation_label.setText('Run enabled: all required inputs are ready.')

    def closeEvent(self, event):
        self._stop_fps_monitor()
        self.shutdown()
        super().closeEvent(event)
