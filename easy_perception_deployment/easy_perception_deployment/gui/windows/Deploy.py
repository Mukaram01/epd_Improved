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
from PySide6.QtCore import QObject, QSize, QThread, QTimer, QElapsedTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QLabel,
                               QMessageBox, QPushButton, QWidget,
                               QDoubleSpinBox, QSpinBox)

from windows.Counting import CountingWindow
from windows.Tracking import TrackingWindow
from windows.job_controller import JobController, JobState

_SCHEMA_IMPORT_ROOT = str(Path(__file__).resolve().parents[2])
if _SCHEMA_IMPORT_ROOT in sys.path:
    sys.path.remove(_SCHEMA_IMPORT_ROOT)
sys.path.insert(0, _SCHEMA_IMPORT_ROOT)
from scripts.cli.config_schema import (  # noqa: E402
    ConfigSchemaError,
    SCHEMA_VERSION,
    migrate_input_topic_config,
    migrate_session_config,
    migrate_usecase_config,
    validate_input_topic_config,
    validate_session_config,
    validate_usecase_config,
)


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
    """FPS monitor running in a daemon Python thread.

    This class intentionally avoids Qt signal/QObject ownership so that no GUI
    object is touched from the worker thread. The GUI polls latest text through
    a QTimer on the main thread.
    """

    def __init__(self, usecase_mode):
        self._usecase_mode = usecase_mode
        self._requested_mode = usecase_mode
        self._node = None
        self._subscription = None
        self._stamps = deque(maxlen=30)
        self._running = True
        self._lock = threading.Lock()
        self._latest_text = 'FPS: -- | Latency: --'
        self._context = None
        self._owns_context = False

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

    def get_latest_text(self):
        with self._lock:
            return self._latest_text

    def _set_latest_text(self, text):
        with self._lock:
            self._latest_text = text

    def _run(self):
        if not _RCLPY_AVAILABLE:
            self._set_latest_text('FPS: N/A | Latency: N/A (ROS unavailable)')
            return
        try:
            if rclpy.ok():
                self._context = rclpy.get_default_context()
            else:
                self._context = rclpy.context.Context()
                self._context.init(args=None)
                self._owns_context = True

            self._node = Node('epd_fps_monitor', context=self._context)
            self._update_subscription(self._usecase_mode)
            while self._running and rclpy.ok(context=self._context):
                self._maybe_update_subscription()
                rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception as exc:
            logging.getLogger('deploy').warning(
                'FPS monitor thread failed: %s', exc)
            self._set_latest_text('FPS: N/A | Latency: N/A (ROS error)')
        finally:
            if self._subscription is not None and self._node is not None:
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
            if self._owns_context and self._context is not None:
                try:
                    self._context.shutdown()
                except Exception as e:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error during context shutdown: %s', e)
                self._context = None

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
            self._set_latest_text('FPS: -- | Latency: --')
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
        self._set_latest_text(f'FPS: {fps_text} | Latency: {latency_text}')

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

        self._DEFAULT_DEPLOY_WIN_H = 540
        self._DEFAULT_DEPLOY_WIN_W = 500
        self._MIN_DEPLOY_WIN_H = 500
        self._MIN_DEPLOY_WIN_W = 420

        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))

        self._deploy_process = None
        self._kill_process = None
        self._deploy_timer = None
        self._kill_timer = None
        self._deploy_start_timeout_timer = None
        self._stop_timeout_timer = None
        self._shutdown_poll_timer = None
        self._shutdown_elapsed = None
        self._shutdown_timeout_ms = 0
        self._deploy_log_file = None
        self._kill_log_file = None
        self._fps_monitor = None
        self._fps_poll_timer = None
        self._is_shutting_down = False
        self._is_app_exiting = False
        self._job_controller = JobController('Deployment', self)
        self._job_controller.state_changed.connect(self._on_job_state_changed)

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
            session_config = migrate_session_config(session_config)
            usecase_config = migrate_usecase_config(usecase_config)
        except ConfigSchemaError as exc:
            self._show_blocking_config_error(str(exc), abort_operation=True)

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
        try:
            image_topic_data = migrate_input_topic_config(image_topic_data)
        except ConfigSchemaError as exc:
            self._show_blocking_config_error(str(exc), abort_operation=True)
        self._write_json_atomic(
            self._path_to_session_config, json.dumps(session_config, indent=4))
        self._write_json_atomic(
            self._path_to_usecase_config, json.dumps(usecase_config, indent=4))
        self._write_json_atomic(
            self._path_to_input_image_json_file, json.dumps(image_topic_data, indent=4))
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
        self.resize(self._DEFAULT_DEPLOY_WIN_W, self._DEFAULT_DEPLOY_WIN_H)
        self.setMinimumSize(self._MIN_DEPLOY_WIN_W, self._MIN_DEPLOY_WIN_H)

        self.setButtons()
        self.fps_label.setText('FPS: disabled for debug')
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
        self.status_label = QLabel(self._job_controller.message, self)
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
        if os.getenv('EPD_DISABLE_FPS_MONITOR') == '1':
            self.fps_label.setText('FPS: disabled')
            return
        if not _RCLPY_AVAILABLE:
            self.fps_label.setText('FPS: N/A | Latency: N/A (ROS unavailable)')
            return

        self._fps_monitor = FPSMonitorThread(self.usecase_mode)
        self._fps_monitor.start()

        self._fps_poll_timer = QTimer(self)
        self._fps_poll_timer.setInterval(250)
        self._fps_poll_timer.timeout.connect(self._poll_fps_monitor)
        self._fps_poll_timer.start()
        self._poll_fps_monitor()

    @Slot()
    def _poll_fps_monitor(self):
        if self._fps_monitor is None:
            return
        self.fps_label.setText(self._fps_monitor.get_latest_text())

    def _update_fps_monitor_mode(self, usecase_mode):
        if self._fps_monitor is not None:
            self._fps_monitor.set_usecase_mode(usecase_mode)

    def _stop_fps_monitor(self):
        if self._fps_poll_timer is not None:
            self._fps_poll_timer.stop()
            self._fps_poll_timer.deleteLater()
            self._fps_poll_timer = None

        if self._fps_monitor is None:
            return
        self._fps_monitor.stop()
        self._fps_monitor.wait(1000)
        self._fps_monitor = None
        self._fps_poll_timer = None

    @Slot(object, str)
    def _on_job_state_changed(self, state, message):
        self.status_label.setText(message)
        is_running = state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING)
        self.run_button.setText('Stop' if is_running else 'Run')
        self.run_button.setIcon(QIcon(
            self._image_path('quit.png' if is_running else 'go.png')))
        self.run_button.setIconSize(QSize(100, 100))
        self.run_button.updateGeometry()
        self.validateDeployInputs()

    def deployPackage(self):
        '''
        A Mutator function that runs a bash script that
        checks the deployment job state.\n
        If False, run bash script to run ROS2 package with
        session_config.json and usecase_config.json
        Otherwise, run bash script to kill ROS2 package
        processes remotely.
        '''
        if self._job_controller.state in (JobState.IDLE, JobState.FAILED):
            self._job_controller.set_state(JobState.STARTING, 'Starting deployment...')
            self._deploy_process, self._deploy_timer = self._start_process(
                [self._deploy_script_path(),
                 str(self.useCPU),
                 str(self.visualizeFlag),
                 '--non-interactive'],
                'deploy',
                cwd=self._scripts_dir())
            if self._deploy_start_timeout_timer is not None:
                self._deploy_start_timeout_timer.stop()
            self._deploy_start_timeout_timer = QTimer(self)
            self._deploy_start_timeout_timer.setSingleShot(True)
            self._deploy_start_timeout_timer.timeout.connect(self._handle_start_timeout)
            self._deploy_start_timeout_timer.start(15000)
        elif self._job_controller.state == JobState.RUNNING:
            self._stop_deployment()

    def _stop_deployment(self):
        if self._job_controller.state == JobState.STOPPING:
            return
        self._job_controller.set_state(JobState.STOPPING, 'Stopping deployment...')
        self.deploy_logger.info("Killing epd_test_container docker.")
        self._kill_process, self._kill_timer = self._start_process(
            [self._kill_script_path()],
            'kill',
            cwd=self._scripts_dir())
        if self._stop_timeout_timer is not None:
            self._stop_timeout_timer.stop()
        self._stop_timeout_timer = QTimer(self)
        self._stop_timeout_timer.setSingleShot(True)
        self._stop_timeout_timer.timeout.connect(self._handle_stop_timeout)
        self._stop_timeout_timer.start(5000)

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
            if process_type == 'deploy' and self._job_controller.state == JobState.STARTING:
                self._job_controller.set_state(JobState.RUNNING, 'Deployment running.')
            return

        timer.stop()
        if process_type == 'deploy':
            self._deploy_timer = None
            self._deploy_process = process
        elif process_type == 'kill':
            self._kill_timer = None
            self._kill_process = process
        self._close_process_log_file(process_type)
        if process_type == 'kill':
            if self._stop_timeout_timer is not None:
                self._stop_timeout_timer.stop()
            kill_result = self._get_kill_status_from_log()
            if process.returncode == 0:
                self._job_controller.set_state(
                    JobState.IDLE,
                    'Stopped' if kill_result == 'STOPPED'
                    else 'Stopped (already stopped)')
                return
            if process.returncode == 2 and kill_result == 'PARTIAL_CLEANUP':
                self._job_controller.set_state(JobState.IDLE, 'Stopped (partial cleanup)')
                self._show_kill_partial_cleanup_warning()
                return

        if process.returncode != 0:
            self._handle_process_error(process_type)
            return

        if process_type == 'kill':
            self._job_controller.set_state(JobState.IDLE, 'Stopped')
        elif process_type == 'deploy':
            self._job_controller.set_state(JobState.RUNNING, 'Deployment running.')

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
        if process_type == 'kill' and self._stop_timeout_timer is not None:
            self._stop_timeout_timer.stop()
        self._job_controller.set_state(JobState.FAILED, 'Error')

        log_tail = self._tail_process_log(process_type)
        error_code = self._extract_epd_error_code(log_tail)
        remediation_text = self._epd_remediation_message(error_code)
        message_lines = [
            f"{process_type.capitalize()} failed.",
        ]
        if error_code:
            message_lines.extend([
                "",
                f"Error code: {error_code}",
            ])
        if remediation_text:
            message_lines.extend([
                "",
                "Suggested remediation:",
                remediation_text
            ])
        message_lines.extend([
            "",
            "log output:",
            log_tail
        ])
        msgBox = QMessageBox()
        msgBox.setText('\n'.join(message_lines))
        msgBox.exec()

    def _extract_epd_error_code(self, log_tail):
        for line in log_tail.splitlines():
            stripped = line.strip()
            if stripped.startswith('EPD_ERR_') and ':' in stripped:
                return stripped.split(':', 1)[0]
        return None

    def _epd_remediation_message(self, error_code):
        remediations = {
            'EPD_ERR_ROS_SETUP_MISSING':
                'ROS 2 setup was not found. Install the configured ROS distro and verify setup.bash exists.',
            'EPD_ERR_DOCKER_NOT_FOUND':
                'Docker command was not found. Install Docker or set EPD_DOCKER_CMD to a valid command.',
            'EPD_ERR_DOCKER_UNAVAILABLE':
                'Docker is not accessible. Add your user to docker group, run with --sudo-docker, or set EPD_DOCKER_CMD.',
            'EPD_ERR_IMAGE_NOT_FOUND':
                'Required EPD image is missing. Pull or build the image and retry deployment.',
            'EPD_ERR_WORKSPACE_MISSING':
                'Workspace path is missing. Ensure EPD_WORKSPACE_ROOT points to a valid workspace directory.',
            'EPD_ERR_WORKSPACE_LAYOUT':
                'deploy.sh is not inside the expected workspace layout. Verify checkout structure and EPD_WORKSPACE_ROOT.',
            'EPD_ERR_VENDOR_MISSING':
                'Mounted workspace is missing epd_onnxruntime_vendor. Verify repository content and mount path.',
            'EPD_ERR_LAUNCH_SCRIPT_INVALID':
                'Container launch script was not found/executable. Check launch.sh/build_launch.sh permissions and paths.'
        }
        return remediations.get(error_code)

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

    def _all_stop_processes_finished(self):
        kill_done = (self._kill_process is None) or (self._kill_process.poll() is not None)
        deploy_done = (self._deploy_process is None) or (self._deploy_process.poll() is not None)
        return kill_done and deploy_done

    def _cleanup_shutdown_resources(self):
        if self._deploy_start_timeout_timer is not None:
            self._deploy_start_timeout_timer.stop()
        if self._stop_timeout_timer is not None:
            self._stop_timeout_timer.stop()
        if self._shutdown_poll_timer is not None:
            self._shutdown_poll_timer.stop()
        if self._deploy_timer is not None and self._deploy_timer.isActive():
            self._deploy_timer.stop()
        if self._kill_timer is not None and self._kill_timer.isActive():
            self._kill_timer.stop()
        self._close_process_log_file('deploy')
        self._close_process_log_file('kill')
        self._deploy_process = None
        self._kill_process = None
        self._deploy_timer = None
        self._kill_timer = None
        self._deploy_start_timeout_timer = None
        self._stop_timeout_timer = None
        self._shutdown_poll_timer = None
        self._shutdown_elapsed = None
        self._shutdown_timeout_ms = 0

    def _finalize_shutdown_cleanup(self):
        self._cleanup_shutdown_resources()
        if self._job_controller.state != JobState.FAILED:
            self._job_controller.set_state(JobState.IDLE, 'Stopped')
        self._is_shutting_down = False

    def shutdown(self):
        self._stop_fps_monitor()

        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        if self._job_controller.state in (JobState.STARTING, JobState.RUNNING):
            self._stop_deployment()

        self._wait_for_graceful_stop(timeout_sec=4)

    def _wait_for_graceful_stop(self, timeout_sec):
        self._shutdown_timeout_ms = int(max(timeout_sec, 0) * 1000)
        self._shutdown_elapsed = QElapsedTimer()
        self._shutdown_elapsed.start()

        if self._shutdown_poll_timer is None:
            self._shutdown_poll_timer = QTimer(self)
            self._shutdown_poll_timer.setInterval(100)
            self._shutdown_poll_timer.timeout.connect(self._on_shutdown_poll_tick)

        if self._all_stop_processes_finished():
            self._finalize_shutdown_cleanup()
            return

        self._shutdown_poll_timer.start()

    def _on_shutdown_poll_tick(self):
        if self._all_stop_processes_finished():
            self._finalize_shutdown_cleanup()
            return

        if self._shutdown_elapsed is None:
            return

        if self._shutdown_elapsed.elapsed() >= self._shutdown_timeout_ms:
            self._handle_stop_timeout()
            self._finalize_shutdown_cleanup()

    def _handle_start_timeout(self):
        if self._job_controller.state != JobState.STARTING:
            return
        if self._deploy_process is not None and self._deploy_process.poll() is None:
            self._job_controller.set_state(
                JobState.RUNNING,
                'Deployment running (startup timeout reached; monitoring logs).')
            return
        self._job_controller.set_state(JobState.FAILED, 'Deployment failed to start in time.')
        self._handle_process_error('deploy')

    def _handle_stop_timeout(self):
        if self._job_controller.state != JobState.STOPPING:
            return
        self.deploy_logger.warning('Stop timeout reached; escalating to terminate/kill.')
        self._terminate_process(self._kill_process, self._kill_timer)
        self._terminate_process(self._deploy_process, self._deploy_timer)
        self._job_controller.set_state(
            JobState.IDLE,
            'Stopped (forced after timeout escalation).')
        if not self.debug and not self._is_app_exiting and not self._is_shutting_down:
            QMessageBox.warning(
                self,
                'Stop Escalated',
                'Graceful stop timed out. Deployment was forcefully terminated.')

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

        dict = {"schema_version": SCHEMA_VERSION, "input_image_topic": new_image_topic}
        try:
            dict = validate_input_topic_config(dict)
        except ConfigSchemaError as exc:
            self._show_blocking_config_error(str(exc))
            return
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

    def _set_usecase_combo_to_mode(self, usecase_mode):
        usecase_name = self._usecase_name_from_mode(usecase_mode)
        usecase_index = self.usecase_config_button.findText(usecase_name)
        if usecase_index >= 0:
            self.usecase_config_button.blockSignals(True)
            self.usecase_config_button.setCurrentIndex(usecase_index)
            self.usecase_config_button.blockSignals(False)

    def _usecase_name_from_mode(self, usecase_mode):
        usecase_names_by_mode = [
            'Classification',
            'Counting',
            'Color-Matching',
            'Localization',
            'Tracking']
        if 0 <= usecase_mode < len(usecase_names_by_mode):
            return usecase_names_by_mode[usecase_mode]
        return usecase_names_by_mode[0]

    def _commit_selected_mode(self, usecase_mode):
        self.usecase_mode = usecase_mode
        self.usecase_config_button.setStyleSheet(
            'background-color: rgba(0,150,10,255);')
        self._update_fps_monitor_mode(self.usecase_mode)

    def setUseCase(self, index):
        '''A function is triggered by the DropDown Menu labelled, UseCase.'''
        previous_usecase_mode = self.usecase_mode
        selected_usecase = self.usecase_list[index]

        if selected_usecase == 'Classification':
            if not self.debug:
                msgBox = QMessageBox()
                msgBox.setText('[Classification] Selected.'
                               'No other configuration required.')
                msgBox.exec()

            self.deploy_logger.info('Wrote to ../data/usecase_config.json')
            dict = {"schema_version": SCHEMA_VERSION, "usecase_mode": 0}
            try:
                dict = validate_usecase_config(dict, require_mode_specific=True)
            except ConfigSchemaError as exc:
                self._show_blocking_config_error(str(exc))
                return
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)
            self._commit_selected_mode(0)

        elif selected_usecase == 'Counting':
            self.counting_window = CountingWindow(self._path_to_label_list,
                                                  self._path_to_usecase_config)
            self.counting_window.show()
            self._commit_selected_mode(1)
        elif selected_usecase == 'Localization':
            dict = {"schema_version": SCHEMA_VERSION, "usecase_mode": 3}
            try:
                dict = validate_usecase_config(dict, require_mode_specific=True)
            except ConfigSchemaError as exc:
                self._show_blocking_config_error(str(exc))
                return
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)
            self._commit_selected_mode(3)
        elif selected_usecase == 'Tracking':
            self.tracking_window = TrackingWindow(self._path_to_usecase_config)
            self.tracking_window.show()
            self._commit_selected_mode(4)
        elif selected_usecase == 'Color-Matching':
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

            if not ok:
                self.deploy_logger.warning('No reference color template set.')
                self._set_usecase_combo_to_mode(previous_usecase_mode)
                return

            resolved_template_path = self.resolveFilePath(input_refimage_filepath)
            if not resolved_template_path or not Path(resolved_template_path).exists():
                self.deploy_logger.warning('No reference color template set.')
                self._set_usecase_combo_to_mode(previous_usecase_mode)
                return

            path_to_color_template = self._normalize_data_path(
                input_refimage_filepath)
            if not path_to_color_template:
                self.deploy_logger.warning('No reference color template set.')
                self._set_usecase_combo_to_mode(previous_usecase_mode)
                return

            self.deploy_logger.info('Wrote to ../data/usecase_config.json')
            dict = {
                "schema_version": SCHEMA_VERSION,
                "usecase_mode": 2,
                "path_to_color_template": path_to_color_template,
                "color_match_histogram_metric": "Correlation"
                }
            try:
                dict = validate_usecase_config(dict, require_mode_specific=True)
            except ConfigSchemaError as exc:
                self._show_blocking_config_error(str(exc))
                self._set_usecase_combo_to_mode(previous_usecase_mode)
                return
            json_object = json.dumps(dict, indent=4)
            self._write_json_atomic(self._path_to_usecase_config, json_object)
            self._commit_selected_mode(2)
        else:
            self.deploy_logger.warning('Invalid Use Case')
            sys.exit()

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
            "schema_version": SCHEMA_VERSION,
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
        try:
            dict = validate_session_config(dict)
        except ConfigSchemaError as exc:
            self._show_blocking_config_error(str(exc))
            return
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
        dict = {
            "schema_version": SCHEMA_VERSION,
            "input_image_topic": self.DEFAULT_INPUT_TOPIC}
        dict = validate_input_topic_config(dict)
        json_object = json.dumps(dict, indent=4)
        self._write_json_atomic(self._path_to_input_image_json_file, json_object)
        self.validateDeployInputs()

    def validateDeployInputs(self):
        '''Validate inputs and update the Run button state.'''
        if self._job_controller.state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING):
            self.run_button.setEnabled(True)
            self.run_button.setToolTip('')
            self.validation_label.setText(self._job_controller.message)
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
        self._is_app_exiting = True
        self.shutdown()
        super().closeEvent(event)
