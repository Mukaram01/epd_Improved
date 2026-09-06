# Copyright 2022 Advanced Remanufacturing and Technology Centre
# Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
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

import signal
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from windows.Main import MainWindow
from windows.acceptance_stability import (
    apply_deploy_ui_stability,
    install_ros_executor_stability,
)
from windows.deploy_runtime_truth import apply_deploy_runtime_truth
from windows.epd0_productization import apply_epd0_productization
from windows.epd1_productization import apply_epd1_productization
from windows.epd2_productization import apply_epd2_productization
from windows.epd3_productization import apply_epd3_productization
from windows.epd4_productization import apply_epd4_productization
from windows.epd5_integration import finalize_epd5_integration
from windows.epd5_productization import apply_epd5_productization
from windows.epd6_productization import apply_epd6_productization
from windows.epd8_productization import apply_epd8_productization
from windows.epd9_productization import apply_epd9_productization

signal.signal(signal.SIGINT, signal.SIG_DFL)
myapp = QApplication(sys.argv)


def main():

    # Install the ROS worker patch before MainWindow constructs the Deploy FPS
    # monitor. This keeps all GUI ROS subscribers off rclpy's global executor.
    install_ros_executor_stability()

    window1 = MainWindow()

    # Acceptance stability replaces the Deploy presentation controller's sync
    # function. Install it before EPD-0 so EPD-0 can wrap the stable sync and
    # remain the final owner of detected/configured camera truth.
    apply_deploy_ui_stability(window1)
    apply_epd0_productization(window1)
    apply_deploy_runtime_truth(window1)
    apply_epd1_productization(window1)
    apply_epd2_productization(window1)
    apply_epd3_productization(window1)
    apply_epd4_productization(window1)
    epd5 = apply_epd5_productization(window1)
    finalize_epd5_integration(window1, epd5)
    apply_epd6_productization(window1)
    apply_epd8_productization(window1)
    apply_epd9_productization(window1)
    window1.help_window.setWindowFlag(Qt.Window, True)
    window1.show()

    myapp.exec()
    sys.exit()


if __name__ == '__main__':
    main()
