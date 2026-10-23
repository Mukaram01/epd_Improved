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

import os
import subprocess
from pathlib import Path


RUN_BASH = Path(__file__).resolve().parent / "run.bash"
WORKSPACE = RUN_BASH.parents[3]


def _resolve(**environment):
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(RUN_BASH), "--print-workspace-setup"],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def test_checkout_workspace_wins_over_unrelated_colcon_prefix():
    assert _resolve(EPD_WS="", COLCON_PREFIX_PATH="/home/example/ws_moveit2/install") == str(
        WORKSPACE / "install" / "setup.bash")


def test_explicit_epd_workspace_override_wins():
    assert _resolve(EPD_WS="/tmp/operator_epd_ws", COLCON_PREFIX_PATH="/tmp/other/install") == (
        "/tmp/operator_epd_ws/install/setup.bash")
