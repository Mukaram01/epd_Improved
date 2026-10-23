# Copyright 2026 Easy Perception Deployment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared pytest setup for the EPD GUI suites.

Two guarantees every GUI test gets:

1. Isolation from the tracked package tree. The windows resolve config/,
   data/, scripts/ and log/ from the package root, and the legacy tests also
   use cwd-relative paths such as ``../config/usecase_config.json``. Each
   test therefore runs inside a throwaway copy of the package with the
   window classes pointed at it, so tracked files stay byte-identical.

2. No modal dialog can block the run. ``QMessageBox`` exec/static helpers
   return immediately and are recorded in the ``message_boxes`` fixture so a
   test can assert that the user would have been told about an error.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from windows.Deploy import DeployWindow
from windows.Main import MainWindow
from windows.Train import TrainWindow

GUI_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = GUI_DIR.parent

# Top-level package entries that GUI tests never read; `test/` holds
# ~25 MB of C++ fixtures.
_SKIPPED_ROOT_ENTRIES = {'test'}
_SKIPPED_NAMES = {'__pycache__', '.pytest_cache', 'log'}

_PRISTINE_ROOT = None

# Captured when conftest loads, before any test module is collected, so these
# are the classes the tests construct. Some tests (test_deploy_import_path.py)
# re-import windows.* against stub Qt modules; patching whatever sys.modules
# holds at fixture time would then miss the real classes and let them write
# into the tracked package tree.
_SANDBOXED_WINDOW_CLASSES = (DeployWindow, MainWindow, TrainWindow)


def _ignore(directory, names):
    ignored = {name for name in names if name in _SKIPPED_NAMES}
    if Path(directory) == PACKAGE_ROOT:
        ignored |= _SKIPPED_ROOT_ENTRIES & set(names)
    return ignored


def pytest_configure(config):
    global _PRISTINE_ROOT
    base = Path(tempfile.mkdtemp(prefix='epd_gui_tests_'))
    _PRISTINE_ROOT = base / 'pristine'
    shutil.copytree(PACKAGE_ROOT, _PRISTINE_ROOT, ignore=_ignore)
    # QSettings (window geometry) must not touch the developer's profile.
    # Set before pytest-qt creates the QApplication.
    os.environ['XDG_CONFIG_HOME'] = str(base / 'xdg_config')
    config._epd_gui_sandbox_base = base


def pytest_unconfigure(config):
    base = getattr(config, '_epd_gui_sandbox_base', None)
    if base is not None:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(autouse=True)
def epd_package_root(tmp_path, monkeypatch):
    """Run the test against a fresh copy of the package tree."""
    root = tmp_path / 'easy_perception_deployment'
    shutil.copytree(_PRISTINE_ROOT, root, symlinks=True)
    gui_dir = root / 'gui'

    for cls in _SANDBOXED_WINDOW_CLASSES:
        monkeypatch.setattr(cls, '_MODULE_DIR', gui_dir / 'windows')
        monkeypatch.setattr(cls, '_GUI_DIR', gui_dir)
        if hasattr(cls, '_PACKAGE_ROOT'):
            monkeypatch.setattr(cls, '_PACKAGE_ROOT', root)

    monkeypatch.chdir(gui_dir)
    return root


class _MessageBoxLog(list):
    def texts(self):
        return [entry['text'] for entry in self]


@pytest.fixture(autouse=True)
def message_boxes(monkeypatch):
    """Record QMessageBox usage instead of opening blocking modal dialogs."""
    log = _MessageBoxLog()

    def _exec(self):
        log.append({
            'kind': 'exec',
            'title': self.windowTitle(),
            'text': self.text(),
        })
        return QMessageBox.Ok

    def _static(kind, default):
        def _show(*args, **kwargs):
            # (parent, title, text, ...) for every static helper.
            title = args[1] if len(args) > 1 else kwargs.get('title', '')
            text = args[2] if len(args) > 2 else kwargs.get('text', '')
            log.append({'kind': kind, 'title': title, 'text': text})
            return default
        return staticmethod(_show)

    monkeypatch.setattr(QMessageBox, 'exec', _exec)
    monkeypatch.setattr(QMessageBox, 'exec_', _exec, raising=False)
    monkeypatch.setattr(
        QMessageBox, 'information', _static('information', QMessageBox.Ok))
    monkeypatch.setattr(
        QMessageBox, 'warning', _static('warning', QMessageBox.Ok))
    monkeypatch.setattr(
        QMessageBox, 'critical', _static('critical', QMessageBox.Ok))
    monkeypatch.setattr(
        QMessageBox, 'question', _static('question', QMessageBox.No))
    return log
