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
"""Shared studio design language for the EPD GUI (applied last, after all productization).

EPD keeps its dark engineering look. This layer aligns it with Workcell Studio (same company,
same token names and dark values as the Builder's studioTokens()) without rewriting windows:

* one application-level base: dark thin scrollbars, tooltips, keyboard focus, split handles;
* the shared accent for primary actions (the old violet was the only off-family colour);
* legacy whole-button green/red readiness paint replaced by a quiet ``ready`` property;
* a readable type floor (no text below 10 px) and an optional text scale (100/112/125 %);
* remembered splitter positions, Reset Layout and a small View menu in the Deploy header.
"""
import re

from PySide6.QtCore import QObject, QEvent, QSettings, Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QApplication, QMenu, QSplitter, QToolButton, QWidget

SETTINGS_ORG = "WorkcellStudio"
SETTINGS_APP = "EasyPerceptionDeployment"
TEXT_SCALE_KEY = "appearance/text_scale"
SPLITTER_KEY = "layout/splitters/"
MIN_FONT_PX = 10

# Workcell Studio dark tokens (workcell_builder_ui_utils.cpp darkTokens()); EPD keeps its own
# backgrounds.
TOKENS = dict(
    border="#2A3647", border_strong="#3B4A5E", text="#E6EDF5", text_secondary="#A9B6C6",
    text_muted="#7F8CA0", accent="#246FE5", accent_hover="#3A7EEB", accent_border="#4C8DFF",
    accent_soft="#1B3159", success="#3DD68C", warning="#F5B14B", danger="#FF7B7B",
    surface="#181D26",
)


def base_stylesheet(t=TOKENS):
    return f"""
    QScrollBar:vertical {{ width: 10px; margin: 2px; background: transparent; border: 0; }}
    QScrollBar:horizontal {{ height: 10px; margin: 2px; background: transparent; border: 0; }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {t['border_strong']}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
    QScrollBar::handle:hover {{ background: {t['text_muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0; height: 0; border: 0; background: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    QToolTip {{ color: #101319; background: {t['text']}; border: 0; padding: 6px 8px;
        border-radius: 4px; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {t['accent_border']}; }}
    QPushButton:focus, QToolButton:focus, QComboBox:focus {{
        border: 1px solid {t['accent_border']}; }}
    """


PRIMARY_RUN_QSS = f"""
    QPushButton#runButton {{ color: #ffffff; background-color: {TOKENS['accent']};
        border: 1px solid {TOKENS['accent_border']}; border-radius: 9px; padding: 9px 18px;
        font-size: 12px; font-weight: 700; }}
    QPushButton#runButton:hover {{ background-color: {TOKENS['accent_hover']}; }}
    QPushButton#runButton:disabled {{
        color: #8a94a6; background-color: #262c37; border-color: #343c4a; }}
    QToolButton#studioViewMenu {{ color: #dce3ed; background-color: #121720;
        border: 1px solid #303a49; border-radius: 7px; padding: 6px 22px 6px 10px;
        font-size: 11px; }}
    QToolButton#studioViewMenu:hover {{ background-color: #1c2330; border-color: #536278; }}
    QToolButton#studioViewMenu::menu-indicator {{ subcontrol-position: right center; right: 7px; }}
    QPushButton[ready="false"] {{ border: 1px solid {TOKENS['danger']}; }}
"""

_FONT_PX = re.compile(r"font-size:\s*(\d+)px")


def scale_font_sizes(qss, scale=1.0, minimum=MIN_FONT_PX):
    """Every ``font-size: Npx`` scaled and floored at ``minimum`` (other rules untouched)."""
    def scaled(match):
        return f"font-size: {max(minimum, round(int(match.group(1)) * scale))}px"
    return _FONT_PX.sub(scaled, qss)


def text_scale(settings=None):
    settings = settings or QSettings(SETTINGS_ORG, SETTINGS_APP)
    try:
        value = float(settings.value(TEXT_SCALE_KEY, 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return min(1.5, max(0.9, value))


def apply_type_scale(windows, scale):
    """Rewrite font sizes of every widget stylesheet under ``windows`` (originals kept)."""
    for window in windows:
        for widget in [window] + window.findChildren(QWidget):
            original = widget.property("studioOriginalQss")
            if original is None:
                original = widget.styleSheet()
                if "font-size" not in original:
                    continue
                widget.setProperty("studioOriginalQss", original)
            widget.setStyleSheet(scale_font_sizes(original, scale))


class SplitterMemory(QObject):
    """Restores and saves the sizes of every named QSplitter under a window."""

    def __init__(self, window, settings):
        super().__init__(window)
        self.settings = settings
        self.splitters = [s for s in window.findChildren(QSplitter) if s.objectName()]
        for splitter in self.splitters:
            saved = settings.value(SPLITTER_KEY + splitter.objectName())
            if saved is not None:
                splitter.restoreState(saved)
            splitter.splitterMoved.connect(lambda *_, s=splitter: self.save(s))

    def save(self, splitter):
        self.settings.setValue(SPLITTER_KEY + splitter.objectName(), splitter.saveState())

    def reset(self):
        for splitter in self.splitters:
            self.settings.remove(SPLITTER_KEY + splitter.objectName())
            count = splitter.count()
            total = sum(splitter.sizes()) or 1
            # Equal-weight default; the splitter's stretch factors settle the rest on resize.
            splitter.setSizes([total // max(1, count)] * count)


def _neutral_ready_style(deploy):
    """Legacy code paints whole buttons green/red; show that state as a quiet property."""
    def set_ready(widget, is_ready):
        widget.setStyleSheet("")
        widget.setProperty("ready", bool(is_ready))
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    deploy._set_ready_style = set_ready
    for widget in (getattr(deploy, "model_button", None), getattr(deploy, "list_button", None),
                   getattr(deploy, "usecase_config_button", None)):
        if widget is not None and "rgba(" in widget.styleSheet():
            set_ready(widget, "0,150,10" in widget.styleSheet().replace(" ", ""))


class _ReadyPaintGuard(QObject):
    """Catches legacy direct green/red setStyleSheet() calls that bypass _set_ready_style()."""

    def __init__(self, deploy):
        super().__init__(deploy)
        self.deploy = deploy
        for widget in (deploy.model_button, deploy.list_button, deploy.usecase_config_button):
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.StyleChange:
            sheet = obj.styleSheet().replace(" ", "")
            if "rgba(0,150,10" in sheet or "rgba(200,10,0" in sheet:
                self.deploy._set_ready_style(obj, "rgba(0,150,10" in sheet)
        return False


def _install_view_menu(main_window, memory_by_window):
    deploy = main_window.deploy_window
    ui = getattr(main_window, "_deploy_ui_controller", None)
    badge = getattr(ui, "header_badge", None)
    header = badge.parentWidget() if badge is not None else None
    if header is None or header.layout() is None:
        return None
    button = QToolButton(deploy)
    button.setObjectName("studioViewMenu")
    button.setText("View")
    button.setToolTip("Text size and layout")
    button.setPopupMode(QToolButton.InstantPopup)
    menu = QMenu(button)
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    windows = [main_window, deploy, main_window.train_window, main_window.help_window]
    size_menu = menu.addMenu("Text size")
    group = QActionGroup(size_menu)
    group.setExclusive(True)
    current = text_scale(settings)
    for label, value in (("100 %", 1.0), ("112 %", 1.125), ("125 %", 1.25)):
        action = size_menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(abs(current - value) < 1e-3)
        group.addAction(action)

        def choose(_checked=False, value=value):
            settings.setValue(TEXT_SCALE_KEY, value)
            apply_type_scale(windows, value)
        action.triggered.connect(choose)
    menu.addSeparator()
    reset = menu.addAction("Reset Layout")
    reset.setShortcut("Ctrl+Shift+0")
    reset.setToolTip("Restore default panel sizes and window sizes; text size is kept.")

    def reset_layout():
        for memory in memory_by_window:
            memory.reset()
        for key in ("deploy_window/size", "train_window/size"):
            settings.remove(key)
        deploy.resize(1100, 760)
    reset.triggered.connect(reset_layout)
    button.setMenu(menu)
    index = max(0, header.layout().count() - 1)
    header.layout().insertWidget(index, button, 0, Qt.AlignTop)
    deploy.studio_view_menu = button
    return button


def _layout_contains(layout, widget):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.widget() is widget:
            return True
        if item.layout() is not None and _layout_contains(item.layout(), widget):
            return True
    return False


def hide_unplaced_widgets(root):
    """Hide unnamed labels/buttons left out of their parent's layout by the layered refreshes.

    Unplaced, they paint at the parent's top-left (behind other widgets) and still take focus.
    Named widgets (deliberately positioned overlays) and children of layout-less containers are
    left alone.
    """
    from PySide6.QtWidgets import QAbstractButton, QLabel
    hidden = 0
    for widget in root.findChildren(QWidget):
        if widget.objectName() or widget.isWindow() or widget.isHidden():
            continue
        if not isinstance(widget, (QLabel, QAbstractButton)):
            continue
        parent = widget.parentWidget()
        if parent is None or parent.layout() is None:
            continue
        if _layout_contains(parent.layout(), widget):
            continue
        widget.hide()
        hidden += 1
    return hidden


def apply_studio_design(main_window):
    """Install the shared base, primary accent, readiness styling, type floor, layout memory."""
    app = QApplication.instance()
    app.setStyleSheet(app.styleSheet() + base_stylesheet())
    deploy = main_window.deploy_window
    deploy.setStyleSheet(deploy.styleSheet() + PRIMARY_RUN_QSS)
    _neutral_ready_style(deploy)
    _ReadyPaintGuard(deploy)
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    memories = [SplitterMemory(w, settings) for w in (deploy, main_window.train_window)]
    _install_view_menu(main_window, memories)
    windows = [main_window, deploy, main_window.train_window, main_window.help_window]
    apply_type_scale(windows, text_scale(settings))
    for window in (deploy, main_window.train_window):
        hide_unplaced_widgets(window)
    return memories
