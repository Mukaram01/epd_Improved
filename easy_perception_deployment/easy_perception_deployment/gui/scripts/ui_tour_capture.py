#!/usr/bin/env python3
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
"""Screenshot tour of the real EPD GUI (launcher, Deploy, Train, Help and every visible tab).

Builds the launcher exactly as main.py does, opens each window at --size, and saves
<tag>_<shot>.png plus <tag>_audit.json: visible buttons/labels whose text does not fit, and
content cut off by a scroll area that cannot scroll horizontally. Read-only: nothing is saved
or launched (no camera, no EPD process).

    python3 scripts/ui_tour_capture.py --out DIR --size 1366x768 --tag 1366x768
(run from the gui directory; set QT_SCALE_FACTOR for scaled runs).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))
os.chdir(GUI_DIR)


def audit(root):
    from PySide6.QtWidgets import QAbstractButton, QLabel, QScrollArea
    from PySide6.QtCore import Qt
    clipped, overflow = [], []
    for widget in root.findChildren(QAbstractButton) + root.findChildren(QLabel):
        if not widget.isVisible() or widget.width() <= 0:
            continue
        text = widget.text()
        if not text or '\n' in text or '<' in text:
            continue
        if isinstance(widget, QLabel) and (
                widget.wordWrap() or not widget.pixmap().isNull()):
            continue
        needed = widget.fontMetrics().horizontalAdvance(text)
        if needed > widget.contentsRect().width() + 2:
            clipped.append(dict(cls=type(widget).__name__, name=widget.objectName(),
                                text=text[:60], width=widget.width(), needed=needed))
    for area in root.findChildren(QScrollArea):
        if not area.isVisible() or not area.widget():
            continue
        if area.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
            continue
        excess = area.widget().minimumSizeHint().width() - area.viewport().width()
        if excess > 1:
            overflow.append(dict(name=area.objectName(), overflow_px=excess))
    return dict(clipped=clipped, overflow=overflow)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', default='1366x768')
    parser.add_argument('--tag', default='epd')
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split('x'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import main as epd_main   # creates the QApplication exactly as the product does
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QTabWidget

    app = QApplication.instance()
    launcher = epd_main.build_main_window()
    report = {}

    def shot(widget, name):
        app.processEvents()
        widget.grab().save(str(out / f'{args.tag}_{name}.png'))
        report[name] = dict(audit(widget), size=[widget.width(), widget.height()],
                            device_pixel_ratio=widget.devicePixelRatioF())

    def tabs_of(widget, prefix):
        n = 0
        for tabs in widget.findChildren(QTabWidget):
            if not tabs.isVisible():
                continue
            for i in range(tabs.count()):
                tabs.setCurrentIndex(i)
                for _ in range(5):
                    app.processEvents()
                n += 1
                name = re.sub('[^A-Za-z0-9]+', '', tabs.tabText(i))[:24]
                shot(widget, f'{prefix}_tab{n:02d}_{name}')
            tabs.setCurrentIndex(0)

    steps = []
    steps.append(lambda: (launcher.resize(min(width, 1100), min(height, 760)), launcher.show()))
    steps.append(lambda: shot(launcher, '01_launcher'))
    steps.append(lambda: (launcher.deployPackage(), launcher.deploy_window.resize(width, height)))
    steps.append(lambda: shot(launcher.deploy_window, '02_deploy'))
    steps.append(lambda: tabs_of(launcher.deploy_window, '03_deploy'))
    steps.append(lambda: (launcher.deploy_window.hide(), launcher.openTrainWindow(),
                          launcher.train_window.resize(width, height)))
    steps.append(lambda: shot(launcher.train_window, '04_train'))
    steps.append(lambda: tabs_of(launcher.train_window, '05_train'))
    steps.append(lambda: (launcher.train_window.hide(), launcher.help_window.show(),
                          launcher.help_window.resize(min(width, 1000), min(height, 760))))
    steps.append(lambda: shot(launcher.help_window, '06_help'))

    def run(index=0):
        if index >= len(steps):
            (out / f'{args.tag}_audit.json').write_text(json.dumps(report, indent=1))
            app.quit()
            return
        steps[index]()
        QTimer.singleShot(1200, lambda: run(index + 1))

    QTimer.singleShot(1500, run)
    app.exec()


if __name__ == '__main__':
    main()
