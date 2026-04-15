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
import json
import logging

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)


class CountingWindow(QWidget):
    '''
    The CountingWindow class is a PySide6
    Graphical User Interface (GUI) window that is called by
    DeployWindow class in order to configure a custom
    Counting use-case and write to usecase_config.json.
    '''
    def __init__(self, path_to_label_list, _path_to_usecase_config):
        '''
        The constructor.
        Sets the size of the window and configurations for usecase_config.
        Checks if the usecase_config.json file exists. If true, configure
        accordingly. Otherwise, assign default values.
        Calls setButtons function to populate window with button.
        '''
        super().__init__()

        self.counting_logger = logging.getLogger('counting')

        self._COUNTING_WIN_H = 340
        self._COUNTING_WIN_W = 500

        self.setWindowIcon(QIcon("img/epd_desktop.png"))

        self._label_list = []
        self._select_list = []
        self._path_to_usecase_config = _path_to_usecase_config

        gui_base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        expanded_path = os.path.expandvars(
            os.path.expanduser(path_to_label_list))
        if os.path.isabs(expanded_path):
            path_to_label_list = os.path.abspath(expanded_path)
        else:
            path_to_label_list = os.path.abspath(
                os.path.join(gui_base_path, expanded_path))

        # Check if label-list file exits.
        if not os.path.exists(path_to_label_list):
            msgBox = QMessageBox()
            msgBox.setText('No label list selected. ' +
                           'Please select a label list.')
            msgBox.exec()
        else:
            try:
                with open(path_to_label_list, encoding='utf-8') as f:
                    self._label_list = [
                        line.strip() for line in f if line.strip()]
            except UnicodeDecodeError:
                self.counting_logger.exception(
                    'Unable to decode label list file %s as UTF-8.',
                    path_to_label_list)
                msgBox = QMessageBox()
                msgBox.setText('Failed to read label list. '
                               'Please use a UTF-8 encoded file.')
                msgBox.exec()
            except OSError:
                self.counting_logger.exception(
                    'Unable to open label list file %s.',
                    path_to_label_list)
                msgBox = QMessageBox()
                msgBox.setText('Failed to read label list. '
                               'Please check the file and try again.')
                msgBox.exec()

            # If label-list file is empty after sanitization.
            if len(self._label_list) == 0:
                msgBox = QMessageBox()
                msgBox.setText('Label list selected is empty.')
                msgBox.exec()

        self.setWindowTitle('Choose which objects to count.')
        self.setGeometry(self._COUNTING_WIN_W, self._COUNTING_WIN_H,
                         self._COUNTING_WIN_W, self._COUNTING_WIN_H)
        self.setFixedSize(self._COUNTING_WIN_W, self._COUNTING_WIN_H)

        self.setButtons()

    def setButtons(self):
        '''A Mutator function that defines all buttons in CountingWindow.'''
        # Label List Menu for showing and adding objects-to-count
        self.label_list_dropdown = QComboBox(self)
        self.label_list_dropdown.setMinimumHeight(40)
        for label in self._label_list:
            self.label_list_dropdown.addItem(label)

        self.label_list_dropdown_label = QLabel(self)
        self.label_list_dropdown_label.setText('Available Objects')

        # Selected List Menu for showing and removing objects-to-count
        self.selected_list_menu = QComboBox(self)
        self.selected_list_menu.setMinimumHeight(40)

        self.selected_list_menu_label = QLabel(self)
        self.selected_list_menu_label.setText('Selected Objects')

        # Finish button to save the stored counting and
        # write to usecase_config.json
        self.finish_button = QPushButton('Finish', self)
        self.finish_button.setIcon(QIcon('img/go.png'))
        self.finish_button.setIconSize(QSize(75, 75))
        # Cancel button to exit USE CASE: COUNTING
        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.setIcon(QIcon('img/quit.png'))
        self.cancel_button.setIconSize(QSize(75, 75))

        grid_layout = QGridLayout()
        grid_layout.addWidget(self.selected_list_menu_label, 0, 0)
        grid_layout.addWidget(self.label_list_dropdown_label, 0, 1)
        grid_layout.addWidget(self.selected_list_menu, 1, 0)
        grid_layout.addWidget(self.label_list_dropdown, 1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.finish_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addLayout(grid_layout)

        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addLayout(button_layout)

        self.finish_button.clicked.connect(self.writeToUseCaseConfig)
        self.cancel_button.clicked.connect(self.closeWindow)
        self.label_list_dropdown.activated.connect(self.addObject)
        self.selected_list_menu.activated.connect(self.removeObject)

    def writeToUseCaseConfig(self):
        '''A function that is triggered by the button labelled, Finish.'''
        self.counting_logger.info('Wrote to ../data/usecase_config.json')

        dict = {
            "usecase_mode": 1,
            "class_list": self._select_list
            }
        json_object = json.dumps(dict, indent=4)

        tmp_path = self._path_to_usecase_config + '.tmp'
        with open(tmp_path, 'w') as outfile:
            outfile.write(json_object)
        os.replace(tmp_path, self._path_to_usecase_config)
        self.close()

    def closeWindow(self):
        '''A function that is triggered by the button labelled, Cancel.'''
        self.close()

    def addObject(self, index):
        '''
        A function that is triggered by the DropDown Menu labelled, Available
        Objects
        '''
        # If selected object is a duplicate, ignore it and show feedback.
        for i in range(0, len(self._select_list)):
            if self._label_list[index] == self._select_list[i]:
                self.counting_logger.warning('Duplicate object detected.')
                self.status_label.setText(
                    '\u26a0 "' + self._label_list[index] + '" is already in the list.')
                return

        self._select_list.append(self._label_list[index])
        self.selected_list_menu.addItem(self._label_list[index])
        self.status_label.setText('')

    def removeObject(self, index):
        '''
        A function that is triggered by the DropDown Menu labelled, Selected
        Objects
        '''
        if len(self._select_list) == 0:
            self.counting_logger.warning('Select list is empty.')
            return

        if index < 0 or index >= len(self._select_list):
            self.counting_logger.warning(
                'Remove index %d out of range for list of length %d.',
                index, len(self._select_list))
            return

        self._select_list.remove(self._select_list[index])
        self.selected_list_menu.removeItem(index)
        self.status_label.setText('')
