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
"""Studio design layer: type floor, readiness styling, unplaced widgets, splitter memory."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget)

from windows import studio_design as sd  # noqa: E402

app = QApplication.instance() or QApplication([])


def test_font_sizes_are_scaled_and_never_below_the_readable_floor():
    qss = "QLabel#a { font-size: 9px; } QLabel#b { font-size: 12px; color: #fff; }"
    expected = "QLabel#a { font-size: 10px; } QLabel#b { font-size: 12px; color: #fff; }"
    assert sd.scale_font_sizes(qss) == expected
    assert "font-size: 15px" in sd.scale_font_sizes(qss, 1.25)
    assert "color: #fff" in sd.scale_font_sizes(qss, 1.25)


def test_type_scale_can_be_reapplied_without_compounding():
    root = QWidget()
    child = QLabel("x", root)
    child.setStyleSheet("QLabel { font-size: 12px; }")
    sd.apply_type_scale([root], 1.25)
    sd.apply_type_scale([root], 1.25)
    assert "font-size: 15px" in child.styleSheet()
    sd.apply_type_scale([root], 1.0)
    assert "font-size: 12px" in child.styleSheet()


def test_unplaced_unnamed_widgets_are_hidden_but_overlays_and_laid_out_widgets_stay():
    page = QWidget()
    layout = QVBoxLayout(page)
    placed = QLabel("placed", page)
    layout.addWidget(placed)
    row = QHBoxLayout()
    nested = QPushButton("nested", page)
    row.addWidget(nested)
    layout.addLayout(row)
    orphan = QLabel("Confidence Threshold", page)
    overlay = QLabel("overlay", page)
    overlay.setObjectName("deliberateOverlay")
    page.show()
    assert sd.hide_unplaced_widgets(page) == 1
    assert orphan.isHidden()
    assert not placed.isHidden() and not nested.isHidden() and not overlay.isHidden()


def test_legacy_green_red_paint_becomes_a_quiet_ready_property():
    class Deploy(QWidget):
        pass
    deploy = Deploy()
    deploy.model_button = QPushButton("model", deploy)
    deploy.list_button = QPushButton("labels", deploy)
    deploy.usecase_config_button = QPushButton("mode", deploy)
    deploy.model_button.setStyleSheet("background-color: rgba(0,150,10,255);")
    deploy.list_button.setStyleSheet("background-color: rgba(200,10,0,255);")
    sd._neutral_ready_style(deploy)
    sd._ReadyPaintGuard(deploy)
    assert deploy.model_button.styleSheet() == "" and deploy.model_button.property("ready") is True
    assert deploy.list_button.property("ready") is False
    # A later legacy repaint is caught too.
    deploy.usecase_config_button.setStyleSheet("background-color: rgba(0,150,10,255);")
    assert deploy.usecase_config_button.styleSheet() == ""
    assert deploy.usecase_config_button.property("ready") is True


def test_named_splitters_are_restored_saved_and_reset(tmp_path):
    settings = QSettings(str(tmp_path / "epd.ini"), QSettings.IniFormat)
    window = QWidget()
    layout = QVBoxLayout(window)
    splitter = QSplitter(Qt.Horizontal, window)
    splitter.setObjectName("monitorMainSplitter")
    for _ in range(2):
        splitter.addWidget(QWidget())
    layout.addWidget(splitter)
    window.resize(800, 400)
    window.show()
    app.processEvents()
    splitter.setSizes([200, 600])
    memory = sd.SplitterMemory(window, settings)
    memory.save(splitter)
    assert settings.contains(sd.SPLITTER_KEY + "monitorMainSplitter")
    splitter.setSizes([600, 200])
    sd.SplitterMemory(window, settings)        # a new session restores the saved state
    assert splitter.sizes()[0] < splitter.sizes()[1]
    memory.reset()
    assert not settings.contains(sd.SPLITTER_KEY + "monitorMainSplitter")
