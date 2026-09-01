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
import glob
import sys
import threading
import subprocess
import logging
from pathlib import Path
import shutil
from ast import literal_eval as make_tuple
import json

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)
from trainer.P2Trainer import P2Trainer
from trainer.P3Trainer import P3Trainer
from windows.job_controller import JobController, JobState

_SCHEMA_IMPORT_ROOT = str(Path(__file__).resolve().parents[2] / 'scripts')
if _SCHEMA_IMPORT_ROOT in sys.path:
    sys.path.remove(_SCHEMA_IMPORT_ROOT)
sys.path.insert(0, _SCHEMA_IMPORT_ROOT)
from cli.config_schema import (  # noqa: E402
    ConfigSchemaError,
    migrate_input_topic_config,
    migrate_session_config,
    migrate_usecase_config,
)


class TrainWindow(QWidget):
    '''
    The TrainWindow class is a PySide6 Graphical User Interface (GUI) window
    that is called by MainWindow class in order to configure a custom training
    session and initiates training for a selected Precision-Level.
    '''
    _MODULE_DIR = Path(__file__).resolve().parent
    _GUI_DIR = _MODULE_DIR.parent
    _PACKAGE_ROOT = _GUI_DIR.parent

    def __init__(self, debug=False):
        '''
        The constructor.
        Sets the size of the window and configurations for a training session.
        Calls setButtons function to populate window with button.
        '''
        super().__init__()

        self.train_logger = logging.getLogger('train')

        self.debug = debug

        self._DEFAULT_TRAIN_WIN_H = 650
        self._DEFAULT_TRAIN_WIN_W = 500
        self._MIN_TRAIN_WIN_H = 560
        self._MIN_TRAIN_WIN_W = 440
        self._ROW_THICKNESS = 100

        self.setWindowIcon(QIcon(self._image_path("epd_desktop.png")))

        self.model_name = ''
        self._model_list = []
        self._label_list = []
        self._precision_level = 2

        self._path_to_dataset = ''
        self._path_to_label_list = ''
        self._is_valid_dataset = False
        self.label_process = None
        self._training_thread = None
        self._active_operation = None

        self._is_model_ready = False
        self._is_dataset_linked = False
        self._is_dataset_labelled = False
        self._is_labellist_linked = False

        self.label_train_process = None
        self.label_val_process = None
        self._readiness_message = 'Training is not ready.'
        self._job_controller = JobController('Training job', self)
        self._job_controller.state_changed.connect(self._on_job_state_changed)
        self._labelme_watchdog = QTimer(self)
        self._labelme_watchdog.setInterval(500)
        self._labelme_watchdog.timeout.connect(self._poll_labelme_process)

        self.max_iteration = 3000
        self.checkpoint_period = 200
        self.test_period = 200
        self.steps = '(1000, 1500, 2000, 2500)'

        self.setWindowTitle('Train')
        self.resize(self._DEFAULT_TRAIN_WIN_W, self._DEFAULT_TRAIN_WIN_H)
        self.setMinimumSize(self._MIN_TRAIN_WIN_W, self._MIN_TRAIN_WIN_H)

        self._validate_existing_epd_configs()
        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in TrainWindow.'''
        font_height = self.fontMetrics().height()
        # Keep controls comfortably clickable across DPI/font scaling.
        control_min_height = max(40, int(font_height * 1.9))  # usability floor
        # Preferred height for primary actions while still allowing growth.
        action_pref_height = max(self._ROW_THICKNESS, int(font_height * 3.6))
        # Preferred icon size; decoupled from button height.
        standard_icon_size = QSize(50, 50)
        emphasis_icon_size = QSize(75, 75)

        def _set_row_button_sizing(button, min_height=control_min_height):
            button.setMinimumHeight(min_height)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.p2_button = QPushButton('P2', self)
        self.p2_button.setMinimumHeight(action_pref_height)
        self.p2_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.p2_button.setStyleSheet(
            'background-color: rgba(180,180,180,255);')

        self.p3_button = QPushButton('P3', self)
        self.p3_button.setMinimumHeight(action_pref_height)
        self.p3_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        # Model dropdown menu to select Precision Level specific model
        self.model_selector = QComboBox(self)
        self.model_selector.setMinimumHeight(control_min_height)
        self.model_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_selector.setStyleSheet('background-color: red;')
        self.populateModelSelector()

        # Labeller button to initiate labelme
        self.label_button = QPushButton('Label Dataset', self)
        self.label_button.setIcon(QIcon(self._image_path('label_labelme.png')))
        self.label_button.setIconSize(standard_icon_size)
        _set_row_button_sizing(self.label_button, action_pref_height)
        self.label_button.setStyleSheet(
            'background-color: rgba(0,200,10,255);')
        if self._precision_level == 1:
            self.label_button.hide()

        self.generate_button = QPushButton('Generate Dataset', self)
        self.generate_button.setIcon(QIcon(self._image_path('label_generate.png')))
        self.generate_button.setIconSize(standard_icon_size)
        _set_row_button_sizing(self.generate_button, action_pref_height)
        self.generate_button.setStyleSheet(
            'background-color: rgba(0,200,10,255);')

        # Labeller button to initiate labelme
        self.validate_button = QPushButton('Validate Training', self)
        self.validate_button.setIcon(QIcon(self._image_path('validate.png')))
        self.validate_button.setIconSize(standard_icon_size)
        _set_row_button_sizing(self.validate_button, action_pref_height)

        # Dataset button to prompt input via FileDialogue
        self.dataset_button = QPushButton('Choose Dataset', self)
        self.dataset_button.setIcon(QIcon(self._image_path('dataset.png')))
        self.dataset_button.setIconSize(standard_icon_size)
        _set_row_button_sizing(self.dataset_button, action_pref_height)
        self.dataset_button.setStyleSheet('background-color: red;')

        # Start Training button to start and display training process
        self.train_button = QPushButton('Train', self)
        self.train_button.setIcon(QIcon(self._image_path('train.png')))
        self.train_button.setIconSize(emphasis_icon_size)
        _set_row_button_sizing(self.train_button, action_pref_height)
        self.train_button.setStyleSheet(
            'background-color: rgba(180,180,180,255);')
        self.train_button.setEnabled(False)

        # Set Label List
        self.list_button = QPushButton('Choose Label List', self)
        self.list_button.setIcon(QIcon(self._image_path('label_list.png')))
        self.list_button.setIconSize(emphasis_icon_size)
        _set_row_button_sizing(self.list_button, action_pref_height)
        self.list_button.setStyleSheet(
            'background-color: rgba(200,10,0,255);')

        self.training_config_label = QLabel(self)
        self.training_config_label.setText('Training Parameters')
        self.training_status_label = QLabel(self)
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setMinimumHeight(control_min_height)
        self.training_status_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        self.maxiter_button = QPushButton('MAX ITERATION', self)
        _set_row_button_sizing(self.maxiter_button, control_min_height)
        self.checkpointp_button = QPushButton('CHECKPOINT PERIOD', self)
        _set_row_button_sizing(self.checkpointp_button, control_min_height)
        self.steps_button = QPushButton('STEPS', self)
        _set_row_button_sizing(self.steps_button, control_min_height)
        self.testp_button = QPushButton('TEST PERIOD', self)
        _set_row_button_sizing(self.testp_button, control_min_height)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(max(8, int(font_height * 0.45)))
        top_layout.addWidget(self.p2_button)
        top_layout.addWidget(self.p3_button)
        top_layout.addStretch(1)
        top_layout.addWidget(self.model_selector)
        top_layout.setStretch(0, 0)
        top_layout.setStretch(1, 0)
        top_layout.setStretch(2, 1)
        top_layout.setStretch(3, 3)

        dataset_layout = QGridLayout()
        dataset_layout.setHorizontalSpacing(max(8, int(font_height * 0.45)))
        dataset_layout.setVerticalSpacing(max(8, int(font_height * 0.45)))
        dataset_layout.addWidget(self.label_button, 0, 0)
        dataset_layout.addWidget(self.generate_button, 0, 1)
        dataset_layout.addWidget(self.dataset_button, 1, 0)
        dataset_layout.addWidget(self.validate_button, 1, 1)
        dataset_layout.setColumnStretch(0, 1)
        dataset_layout.setColumnStretch(1, 1)

        training_params_layout = QGridLayout()
        training_params_layout.setHorizontalSpacing(max(8, int(font_height * 0.45)))
        training_params_layout.setVerticalSpacing(max(8, int(font_height * 0.45)))
        training_params_layout.addWidget(self.maxiter_button, 0, 0)
        training_params_layout.addWidget(self.checkpointp_button, 0, 1)
        training_params_layout.addWidget(self.steps_button, 1, 0)
        training_params_layout.addWidget(self.testp_button, 1, 1)
        training_params_layout.setColumnStretch(0, 1)
        training_params_layout.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout_margin = max(10, int(font_height * 0.6))
        layout_spacing = max(8, int(font_height * 0.5))
        layout.setContentsMargins(layout_margin, layout_margin,
                                  layout_margin, layout_margin)
        layout.setSpacing(layout_spacing)
        layout.addLayout(top_layout)
        layout.addWidget(self.list_button)
        layout.addLayout(dataset_layout)
        layout.addWidget(self.training_config_label,
                         alignment=Qt.AlignHCenter)
        layout.addLayout(training_params_layout)
        layout.addWidget(self.train_button)
        layout.addWidget(self.training_status_label)
        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        layout.setStretch(3, 0)
        layout.setStretch(4, 0)
        layout.setStretch(5, 0)
        layout.setStretch(6, 1)

        self.p2_button.clicked.connect(self.setP2)
        self.p3_button.clicked.connect(self.setP3)

        self.model_selector.activated.connect(self.setModel)
        self.dataset_button.clicked.connect(self.setDataset)
        self.label_button.clicked.connect(self.runLabelme)
        self.generate_button.clicked.connect(self.conformDatasetToCOCO)
        self.validate_button.clicked.connect(self.validateTraining)
        self.list_button.clicked.connect(self.setLabelList)

        self.maxiter_button.clicked.connect(self.setMaxIteration)
        self.checkpointp_button.clicked.connect(self.setCheckPointPeriod)
        self.testp_button.clicked.connect(self.setTestPeriod)
        self.steps_button.clicked.connect(self.setSteps)
        self.train_button.clicked.connect(self.updateBeforeStartingTraining)
        self.update_training_readiness()

    def setP2(self):
        '''A function that is triggered by the button labelled, P2.'''
        self._precision_level = 2
        self.populateModelSelector()
        self.setModel(0)
        self.label_button.show()
        self.generate_button.show()
        self.p2_button.setStyleSheet(
            'background-color: rgba(180,180,180,255);')
        self.p3_button.setStyleSheet('background-color: white;')
        self.update_training_readiness()

    def setP3(self):
        '''A function that is triggered by the button labelled, P3.'''
        self._precision_level = 3
        self.populateModelSelector()
        self.setModel(0)
        self.label_button.show()
        self.generate_button.show()
        self.p3_button.setStyleSheet(
            'background-color: rgba(180,180,180,255);')
        self.p2_button.setStyleSheet('background-color: white;')
        self.update_training_readiness()

    def setModel(self, index):
        '''A function that is triggered by
        the DropDown Menu option, Model.
        '''
        self.model_name = self.model_selector.itemText(index)

        self.model_selector.setStyleSheet(
            'background-color: rgba(0,200,10,255);')
        self._is_model_ready = True
        self.train_logger.info('Setting Model to ' + self.model_name)
        self.update_training_readiness()

    def setLabelList(self):
        '''A function that is triggered by Choose Label List button.'''
        if not self.debug:
            input_classes_filepath, ok = (
                QFileDialog.getOpenFileName(
                    self,
                    'Set the .txt to use',
                    str(self._data_dir()),
                    'Text Files (*.txt)'))
        else:
            input_classes_filepath = str(
                self._data_dir() / 'label_list' / 'coco_classes.txt')
            ok = True

        if ok:
            self._path_to_label_list = input_classes_filepath
            with open(input_classes_filepath, encoding='utf-8') as class_file:
                self._label_list = [
                    line.rstrip('\n') for line in class_file]
        else:
            self.train_logger.warning('No label list set.')
            self._is_labellist_linked = False
            self.training_status_label.setText(
                'Label list was not selected. Choose Label List to continue.')
            self.update_training_readiness()
            return
        self.list_button.setStyleSheet('background-color: rgba(0,150,10,255);')
        self._is_labellist_linked = True
        self.update_training_readiness()

    def setDataset(self):
        '''A function that is triggered by Choose Dataset button.'''
        if not self.debug:
            new_filepath_to_dataset = (
                QFileDialog.getExistingDirectory(
                    self,
                    'Set directory of the dataset',
                    str(self._data_dir()),
                    QFileDialog.ShowDirsOnly
                    | QFileDialog.
                    DontResolveSymlinks))

        else:
            new_filepath_to_dataset = str(self._data_dir() / 'datasets')

        if not new_filepath_to_dataset:
            self.train_logger.warning('No dataset directory selected.')
            self._is_dataset_linked = False
            self._is_dataset_labelled = False
            self.dataset_button.setIcon(QIcon(self._image_path('dataset.png')))
            self.dataset_button.setStyleSheet('background-color: red;')
            self.training_status_label.setText(
                'Dataset directory was not selected. '
                'Choose Dataset to continue.')
            self.update_training_readiness()
            return

        self._path_to_dataset = new_filepath_to_dataset
        self.validateDataset(new_filepath_to_dataset)

    def setMaxIteration(self):
        if not self.debug:
            max_iteration, ok = QInputDialog().getInt(
                self,
                "MAX ITERATION",
                "No. of Training Epochs:",
                self.max_iteration)
        else:
            ok = True
            max_iteration = self.max_iteration
        if ok:
            self.max_iteration = max_iteration
            self.train_logger.info(
                "Setting Max Iteration to " + str(self.max_iteration))

    def setCheckPointPeriod(self):
        if not self.debug:
            checkpoint_p, ok = QInputDialog().getInt(
                self,
                "CHECKPOINT PERIOD",
                "Interval to Save Model:",
                self.checkpoint_period)
        else:
            ok = True
            checkpoint_p = self.checkpoint_period
        if ok:
            self.checkpoint_period = checkpoint_p
            self.train_logger.info(
                "Setting Checkpoint Period to " + str(self.checkpoint_period))

    def setTestPeriod(self):
        if not self.debug:
            test_p, ok = QInputDialog().getInt(
                self,
                "TEST PERIOD",
                "Interval to Test Model:",
                self.test_period)
        else:
            ok = True
            test_p = self.test_period
        if ok:
            self.test_period = test_p
            self.train_logger.info(
                "Setting Test Period to " + str(self.test_period))

    def setSteps(self):
        if not self.debug:
            steps, ok = QInputDialog().getText(
                self,
                "STEPS",
                "Tuple of Epochs to Decrease " +
                "Learning Rate Eg. (1000, 2000):",
                QLineEdit.Normal,
                self.steps)
        else:
            ok = False
            steps = self.steps

        if ok:
            # Check if input is a valid tuple
            try:
                local_steps = make_tuple(steps)
            except (ValueError, SyntaxError):
                self.train_logger.exception(
                    "Invalid tuple given. " +
                    "ValueError or SyntaxError detected" +
                    "Assigning default values")
            self.steps = steps
            self.train_logger.info("Setting Steps to " + str(self.steps))

    def runLabelme(self):
        '''A function that is triggered by Label Dataset button.'''
        if self._job_controller.state == JobState.RUNNING and self.label_process is not None:
            self._active_operation = 'labelme'
            self._job_controller.set_state(JobState.STOPPING, 'Stopping labelme...')
            self._terminate_labelme()
            self._active_operation = None
            self._job_controller.set_state(JobState.IDLE, 'Labelme stopped.')
            return

        if self._job_controller.is_busy():
            return

        try:
            self._job_controller.set_state(JobState.STARTING, 'Starting labelme...')
            self._active_operation = 'labelme'
            self.label_process = subprocess.Popen(['labelme'])
            self._labelme_watchdog.start()
            self._job_controller.set_state(JobState.RUNNING, 'Labelme running.')
        except OSError as exc:
            self._active_operation = None
            self._job_controller.set_state(JobState.FAILED, f'Failed to launch labelme: {exc}')
            QMessageBox.warning(self, 'Labelme Failed', str(exc))

    def validateTraining(self):
        '''
        A Mutator function that evaulates necessary boolean flags
        that are all required to allow proper training given
        a certain requested precision level.
        '''
        self._job_controller.set_state(JobState.STARTING, 'Validating training prerequisites...')
        self._active_operation = 'validation'
        unmet = self.update_training_readiness()
        if unmet:
            self.validate_button.setStyleSheet(
                'background-color: rgba(255,255,255,255);')
            self.train_logger.warning(
                'Training validation failed. Unmet prerequisites: ' +
                ', '.join(unmet))
            self._job_controller.set_state(
                JobState.FAILED,
                'Validation failed: missing ' + ', '.join(unmet) + '.')
            self._active_operation = None
            return

        self.validate_button.setStyleSheet(
            'background-color: rgba(0,200,10,255);')
        self.train_logger.info(
            "[ SUCCESS ] - Training Validated. Train button unlocked.")
        self._active_operation = None
        self._job_controller.set_state(JobState.IDLE, 'Validation successful. Ready to train.')

    def update_training_readiness(self):
        unmet = []
        if not self._is_model_ready:
            unmet.append('model')
        if not self._is_dataset_linked:
            unmet.append('dataset')
        if not self._is_labellist_linked:
            unmet.append('label list')
        if not self._is_dataset_labelled:
            unmet.append('labelled dataset')

        if unmet:
            self._readiness_message = (
                'Training is not ready. Missing: ' +
                ', '.join(unmet) +
                '.\nNext step: provide ' + unmet[0] + '.')
            self._update_controls_for_state()
            return unmet

        self._readiness_message = (
            'Training is ready. Click Validate Training or Train to proceed.')
        self._update_controls_for_state()
        return []

    def validateDataset(self, new_filepath_to_dataset):
        isDatasetNamedRight = (
            os.path.basename(self._path_to_dataset)
            == 'custom_dataset')
        PATH_TO_TRAIN_DATASET = (
            self._path_to_dataset + '/train_dataset')
        PATH_TO_VAL_DATASET = (
            self._path_to_dataset + '/val_dataset')
        PATH_TO_TRAIN_ANNOTATIONS = (
            PATH_TO_TRAIN_DATASET + '/annotations.json')
        PATH_TO_VAL_ANNOTATIONS = (
            PATH_TO_VAL_DATASET + '/annotations.json')
        trainDirExists = os.path.exists(PATH_TO_TRAIN_DATASET)
        valDirExists = os.path.exists(PATH_TO_VAL_DATASET)
        annotationsExists = (
            os.path.exists(PATH_TO_TRAIN_ANNOTATIONS) and
            os.path.exists(PATH_TO_VAL_ANNOTATIONS))

        self._is_dataset_labelled = True

        if not trainDirExists:
            self._is_dataset_labelled = False
            self.train_logger.error('Invalid Training Dataset. ' +
                                    '/train_dataset directory MISSING')
        if not valDirExists:
            self._is_dataset_labelled = False
            self.train_logger.error('Invalid Training Dataset. ' +
                                    '/val_dataset directory MISSING')
        if not isDatasetNamedRight:
            self._is_dataset_labelled = False
            self.train_logger.error(
                'Invalid Training Dataset. ' +
                'Dataset folder is not named custom_dataset.')
        if not annotationsExists:
            self._is_dataset_labelled = False
            self.train_logger.error(
                'Invalid Training Dataset. ' +
                'annotations.json files are missing for either' +
                '/train_dataset or /val_dataset.')

        if self._is_dataset_labelled is True:
            self.train_logger.info('[ SUCCESS ] - Training Dataset VALID.')
            # Set button color to green
            self._is_dataset_linked = True
            self.dataset_button.setIcon(QIcon(self._image_path('valid_dataset.png')))
            self.dataset_button.setStyleSheet(
                'background-color: rgba(0,200,10,255);')
        else:
            # Set button color to red
            self.train_logger.warning(
                'Invalid Dataset. Please choose another.')
            self._is_dataset_linked = False
            self.dataset_button.setIcon(QIcon(self._image_path('dataset.png')))
            self.dataset_button.setStyleSheet('background-color: red;')
        self.update_training_readiness()

    def startTraining(self):
        if self._precision_level == 1:
            self.train_logger.warning(
                "[ Deprecation Notice ] - Precision Level 1 features " +
                "has been deprecated in EPD v0.3.0.")
            self.train_logger.warning(
                "Please use Precision Level 1 and 2 features instead.")
            return
        if self._precision_level == 2:
            p2_trainer = P2Trainer(self._path_to_dataset,
                                   self.model_name,
                                   self._label_list,
                                   self.max_iteration,
                                   self.checkpoint_period,
                                   self.test_period,
                                   self.steps)
            p2_trainer.train(False)
            p2_trainer.export(False)
            return

        p3_trainer = P3Trainer(self._path_to_dataset,
                               self.model_name,
                               self._label_list,
                               self.max_iteration,
                               self.checkpoint_period,
                               self.test_period,
                               self.steps)
        p3_trainer.train(False)
        p3_trainer.export(False)

    def updateBeforeStartingTraining(self):
        '''A function that is triggered by the button labelled, Train.'''
        if self._job_controller.is_busy():
            return

        unmet = self.update_training_readiness()
        if unmet:
            return

        self._job_controller.set_state(JobState.STARTING, 'Preparing training job...')
        self._active_operation = 'training'
        self.validate_button.setStyleSheet(
            'background-color: rgba(255,255,255,255);')
        self.validate_button.updateGeometry()
        self._job_controller.set_state(JobState.RUNNING, 'Training in progress. Observe terminal output.')

        d = threading.Thread(name='startTraining', target=self._run_training_job)
        d.setDaemon(True)
        d.start()
        self._training_thread = d

    def _run_training_job(self):
        try:
            self.startTraining()
            self._active_operation = None
            self._job_controller.set_state(JobState.IDLE, 'Training finished.')
        except Exception as exc:
            self._active_operation = None
            self.train_logger.exception('Training failed: %s', exc)
            self._job_controller.set_state(JobState.FAILED, f'Training failed: {exc}')

    def conformDatasetToCOCO(self):
        '''A function that is triggered by Generate Dataset button.'''
        if not self.debug:
            path_to_labelled = (
                QFileDialog.getExistingDirectory(
                    self,
                    'Select your labeled dataset.',
                    str(self._data_dir()),
                    QFileDialog.ShowDirsOnly
                    | QFileDialog.DontResolveSymlinks))
        else:
            self.train_logger.warning('No valid input. Please try again.')
            return

        PATH_TO_TRAIN_DATASET = path_to_labelled + '/train_dataset'
        PATH_TO_VAL_DATASET = path_to_labelled + '/val_dataset'
        # Check if /train_dataset and /val_dataset folders exists
        # in user-provided annotated dataset.
        trainDirExists = os.path.exists(PATH_TO_TRAIN_DATASET)
        if not trainDirExists:
            self.train_logger.error('/train_dataset MISSING.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'Selected folder is missing /train_dataset.')
            return
        valDirExists = os.path.exists(PATH_TO_VAL_DATASET)
        if not valDirExists:
            self.train_logger.error('/val_dataset MISSING.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'Selected folder is missing /val_dataset.')
            return
        # Check if there are non-zero images in /train_dataset.
        no_of_train_image = 0
        for file in os.listdir(PATH_TO_TRAIN_DATASET):
            # Check only .png or .jpeg files.
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()
            if (file_extension == '.png' or
               file_extension == '.jpeg' or
               file_extension == '.jpg'):
                no_of_train_image = no_of_train_image + 1
        doesTrainImagesExists = (no_of_train_image != 0)
        if not doesTrainImagesExists:
            self.train_logger.error('No train images ' +
                                    'found in /train_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'No train images found in /train_dataset.')
            return
        # Check if there are non-zero json files in /train_dataset.
        # Check if there are non-zero images in /train_dataset.
        no_of_train_json = 0
        for file in os.listdir(PATH_TO_TRAIN_DATASET):
            # Check only .json files.
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()
            if file_extension == '.json':
                no_of_train_json = no_of_train_json + 1
        doesTrainJsonsExists = (no_of_train_json != 0)
        if not doesTrainJsonsExists:
            self.train_logger.error('No json files ' +
                                    'found in /train_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'No json files found in /train_dataset.')
            return
        # Check if there are non-zero images in /val_dataset.
        no_of_val_image = 0
        for file in os.listdir(PATH_TO_VAL_DATASET):
            # Check only .png or .jpeg files.
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()
            if (file_extension == '.png' or
               file_extension == '.jpeg' or
               file_extension == '.jpg'):
                no_of_val_image = no_of_val_image + 1
        doesValImagesExists = (no_of_val_image != 0)
        if not doesValImagesExists:
            self.train_logger.error('No val images ' +
                                    'found in /val_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'No validation images found in /val_dataset.')
            return
        # Check if there are non-zero json files in /val_dataset.
        no_of_val_json = 0
        for file in os.listdir(PATH_TO_VAL_DATASET):
            # Check only .json files.
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()
            if file_extension == '.json':
                no_of_val_json = no_of_val_json + 1
        doesValJsonsExists = (no_of_val_json != 0)
        if not doesValJsonsExists:
            self.train_logger.error('No json files ' +
                                    'found in /val_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'No json files found in /val_dataset.')
            return
        # Check if there is a corresponding number of .json
        # and image files in /train_dataset and /val_dataset
        isAnnotatedTrainImageInvalid = (
            no_of_train_image == no_of_train_json)
        if not isAnnotatedTrainImageInvalid:
            self.train_logger.error('Unequal images & .json files ' +
                                    'found in /train_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'Unequal images and json files in /train_dataset.')
            return
        isAnnotatedValImageInvalid = (
            no_of_val_image == no_of_val_json)
        if not isAnnotatedValImageInvalid:
            self.train_logger.error('Unequal images & .json files ' +
                                    'found in /val_dataset.')
            QMessageBox.warning(
                self,
                'Invalid Dataset',
                'Unequal images and json files in /val_dataset.')
            return

        outputTrainDir = str(self._data_dir() / 'datasets' / 'custom_dataset' / 'train_dataset')
        outputValDir = str(self._data_dir() / 'datasets' / 'custom_dataset' / 'val_dataset')
        custom_dataset_dir = self._data_dir() / 'datasets' / 'custom_dataset'

        if self._path_to_label_list == '':
            self.train_logger.error('No Label List provided. ' +
                                    'Please choose Label List.')
            self.training_status_label.setText(
                'No label list provided. '
                'Choose Label List before generating dataset.')
            self.update_training_readiness()
            return

        if custom_dataset_dir.exists():
            self.train_logger.warning('Pre-existing /custom_dataset FOUND.')
            if not self.debug:
                reply = QMessageBox.question(
                    self,
                    'Overwrite Dataset',
                    'A pre-existing custom_dataset directory was found.\n'
                    'Overwrite it?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
                if reply != QMessageBox.Yes:
                    self.train_logger.info('Dataset overwrite cancelled.')
                    self.training_status_label.setText(
                        'Dataset generation cancelled. '
                        'Existing custom_dataset was kept.')
                    self.update_training_readiness()
                    return
            try:
                shutil.rmtree(custom_dataset_dir, ignore_errors=False)
            except FileNotFoundError:
                self.train_logger.warning(
                    'Pre-existing custom_dataset was already removed: %s',
                    custom_dataset_dir)
            except OSError as e:
                self.train_logger.error(
                    'Failed to remove pre-existing custom_dataset %s: %s',
                    custom_dataset_dir, e)
                QMessageBox.warning(
                    self,
                    'Dataset Generation Failed',
                    'Failed to remove existing custom_dataset folder.')
                return

        train_cmd = [
            sys.executable,
            str(self._dataset_script_path()),
            '--labels',
            self._path_to_label_list,
            PATH_TO_TRAIN_DATASET,
            outputTrainDir]
        val_cmd = [
            sys.executable,
            str(self._dataset_script_path()),
            '--labels',
            self._path_to_label_list,
            PATH_TO_VAL_DATASET,
            outputValDir]

        if not self._run_dataset_conversion(
                train_cmd, context='train_dataset conversion'):
            return
        if not self._run_dataset_conversion(
                val_cmd, context='val_dataset conversion'):
            return
        self._path_to_dataset = str(custom_dataset_dir)
        self.validateDataset(self._path_to_dataset)
        self.training_status_label.setText(
            'Dataset conversion complete. '
            'Validate training readiness before training.')
        self.update_training_readiness()

    def _run_dataset_conversion(self, cmd, context, timeout_sec=300):
        self.train_logger.info('Running dataset conversion [%s]: %s', context, cmd)
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self._GUI_DIR),
                check=True,
                timeout=timeout_sec,
                capture_output=True,
                text=True)
        except subprocess.TimeoutExpired:
            self.train_logger.error(
                'Dataset conversion timed out [%s] command=%s timeout=%ss',
                context, cmd, timeout_sec)
            QMessageBox.warning(
                self,
                'Dataset Generation Failed',
                f'Dataset generation timed out during {context}.')
            return False
        except subprocess.CalledProcessError as e:
            self.train_logger.error(
                'Dataset conversion failed [%s] command=%s returncode=%s stderr=%s',
                context, cmd, e.returncode, e.stderr)
            QMessageBox.warning(
                self,
                'Dataset Generation Failed',
                f'Dataset generation failed during {context}.')
            return False
        except OSError as e:
            self.train_logger.error(
                'Failed to invoke dataset conversion [%s] command=%s: %s',
                context, cmd, e)
            QMessageBox.warning(
                self,
                'Dataset Generation Failed',
                f'Failed to run dataset generation during {context}.')
            return False

        self.train_logger.debug(
            'Dataset conversion completed [%s] returncode=%s',
            context, completed.returncode)
        return True

    def populateModelSelector(self):
        '''
        A Mutator function that populates the DropDown Menu labelled, Choose
        Model with all available pretrained models from PyTorch model zoo.
        '''
        # Implement different model list based on different precision level.
        list_path = ''
        if self._precision_level == 2:
            list_path = self._list_path('p2_model_list.txt')
        elif self._precision_level == 3:
            list_path = self._list_path('p3_model_list.txt')

        if list_path:
            try:
                self._model_list = [
                    line.rstrip('\n') for line in open(list_path, encoding='utf-8')]
            except FileNotFoundError:
                self.train_logger.error(
                    'Model list file not found: ' + list_path)
                self._model_list = []
                QMessageBox.warning(
                    self,
                    'Model List Missing',
                    'Model list file not found:\n' + str(list_path) +
                    '\n\nPlease ensure the lists/ directory exists.')

        self.model_selector.clear()
        for model in self._model_list:
            self.model_selector.addItem(model)

        if not self._model_list:
            self.model_selector.setStyleSheet('background-color: red;')
            self._is_model_ready = False

    def _data_dir(self):
        return self._PACKAGE_ROOT / 'data'

    def _validate_existing_epd_configs(self):
        """Validate/migrate deploy config files so Train and Deploy share schema behavior."""
        config_dir = self._PACKAGE_ROOT / 'config'
        validators = [
            ('session_config.json', migrate_session_config),
            ('usecase_config.json', migrate_usecase_config),
            ('input_image_topic.json', migrate_input_topic_config),
        ]
        for filename, validator in validators:
            path = config_dir / filename
            if not path.exists():
                continue
            try:
                with open(path, encoding='utf-8') as infile:
                    data = json.load(infile)
                migrated = validator(data)
                with open(path, 'w', encoding='utf-8') as outfile:
                    json.dump(migrated, outfile, indent=4)
            except (json.JSONDecodeError, OSError, ConfigSchemaError) as exc:
                self.train_logger.warning(
                    'Skipping config validation for %s: %s',
                    filename,
                    exc)

    def _image_path(self, image_name):
        return str(self._GUI_DIR / 'img' / image_name)

    def _list_path(self, filename):
        return str(self._GUI_DIR / 'lists' / filename)

    def _dataset_script_path(self):
        return (self._GUI_DIR / 'dataset' / 'labelme2coco.py').resolve()

    def _terminate_labelme(self):
        if self.label_process is None:
            self._labelme_watchdog.stop()
            return True
        if self.label_process.poll() is not None:
            self.label_process = None
            self._labelme_watchdog.stop()
            return True
        self.label_process.terminate()
        try:
            self.label_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.label_process.kill()
            self.label_process.wait(timeout=1)
        self.label_process = None
        self._labelme_watchdog.stop()
        return True

    def _poll_labelme_process(self):
        if self.label_process is None:
            self._labelme_watchdog.stop()
            return
        if self.label_process.poll() is None:
            return
        self.label_process = None
        self._active_operation = None
        self._job_controller.set_state(JobState.IDLE, 'Labelme exited.')
        self._labelme_watchdog.stop()

    def _update_controls_for_state(self):
        state = self._job_controller.state
        is_busy = state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING)
        unmet = self._training_unmet_prerequisites()

        can_train = (not is_busy) and (not unmet)
        self.train_button.setEnabled(can_train)
        self.train_button.setStyleSheet(
            'background-color: rgba(255,255,255,255);'
            if can_train else 'background-color: rgba(180,180,180,255);')
        self.train_button.setText(
            'Training In Progress. Observe Terminal.' if state == JobState.RUNNING
            else 'Train')
        self.label_button.setEnabled((not is_busy) or self._active_operation == 'labelme')
        self.validate_button.setEnabled(not is_busy)

        if state == JobState.RUNNING and self.label_process is not None:
            self.label_button.setText('Stop Label Dataset')
        else:
            self.label_button.setText('Label Dataset')

        if is_busy or state == JobState.FAILED:
            self.training_status_label.setText(self._job_controller.message)
        else:
            self.training_status_label.setText(self._readiness_message)

    def _training_unmet_prerequisites(self):
        unmet = []
        if not self._is_model_ready:
            unmet.append('model')
        if not self._is_dataset_linked:
            unmet.append('dataset')
        if not self._is_labellist_linked:
            unmet.append('label list')
        if not self._is_dataset_labelled:
            unmet.append('labelled dataset')
        return unmet

    def _on_job_state_changed(self, _state, _message):
        self._update_controls_for_state()

    def closeEvent(self, event):
        close_message = None
        if self.label_process is not None and self.label_process.poll() is None:
            self._job_controller.set_state(JobState.STOPPING, 'Closing window: stopping labelme...')
            self._terminate_labelme()
            close_message = 'Labelme stopped during shutdown.'

        if self._training_thread is not None and self._training_thread.is_alive():
            self._job_controller.set_state(
                JobState.STOPPING,
                'Closing window: waiting for training thread to finish...')
            self._training_thread.join(timeout=5)
            if self._training_thread.is_alive():
                close_message = (
                    'Training thread is still running in background. '
                    'Close completed without forced trainer termination.')
            else:
                close_message = 'Training thread completed during shutdown.'

        if self._job_controller.state != JobState.FAILED:
            self._job_controller.set_state(JobState.IDLE, close_message or 'Window closed.')

        if close_message and not self.debug:
            QMessageBox.information(self, 'Train Window Shutdown', close_message)
        super().closeEvent(event)
