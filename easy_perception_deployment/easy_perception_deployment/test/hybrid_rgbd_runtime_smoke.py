#!/usr/bin/env python3
# Copyright 2026 Advanced Remanufacturing and Technology Centre
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

"""Exercise actual P3 ROS output on a fixture RGB and analytic depth ramp.

These synthetic depth arrays test interfaces/units only, not perception accuracy.
Run in an isolated ROS_DOMAIN_ID with no EMD or robot nodes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def run(binary, image, mode, log_dir):
    tracking = mode == 'tracking'
    args = [str(binary), '--ros-args', '-p', f'usecase_mode_override:={4 if tracking else 3}',
            '-p', 'rgb_topic:=/smoke/rgb', '-p', 'depth_topic:=/smoke/depth',
            '-p', 'camera_info_topic:=/smoke/info']
    if mode != 'default':
        args += ['-p', 'hybrid_refinement_enabled:=' +
                 ('false' if mode == 'disabled' else 'true')]
    output = []
    node = rclpy.create_node('hybrid_smoke')
    bridge = CvBridge()
    pub_rgb = node.create_publisher(Image, '/smoke/rgb', qos_profile_sensor_data)
    pub_depth = node.create_publisher(Image, '/smoke/depth', qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, '/smoke/info', qos_profile_sensor_data)
    sub = node.create_subscription(
        EPDObjectTracking if tracking else EPDObjectLocalization,
        '/easy_perception_deployment/' +
        ('epd_tracking_output' if tracking else 'epd_localize_output'),
        lambda msg: output.append(msg) if msg.objects else None, qos_profile_sensor_data)
    rgb = cv2.resize(cv2.imread(str(image)), (320, 240))
    depth = np.repeat((500 + np.arange(320, dtype=np.uint16))[None, :], 240, axis=0)
    if mode == 'float':
        depth = depth.astype(np.float32) * 0.001
    info = CameraInfo(width=320, height=240)
    info.k = [305., 0., 159.5, 0., 305., 119.5, 0., 0., 1.]
    with (log_dir / (mode + '.log')).open('w') as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "EPD_EXECUTION_BACKEND": "cpu"},
                                   cwd=image.resolve().parent.parent)
        try:
            deadline = time.monotonic() + 45
            sequence = 0
            while len(output) < (2 if tracking else 1) and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'{mode}: node exited {process.returncode}')
                sequence += 1
                msg = bridge.cv2_to_imgmsg(rgb, 'bgr8')
                dep = bridge.cv2_to_imgmsg(depth, '32FC1' if mode == 'float' else '16UC1')
                msg.header.frame_id = 'smoke_optical'
                msg.header.stamp.sec = sequence
                dep.header = info.header = msg.header
                pub_rgb.publish(msg)
                pub_depth.publish(dep)
                pub_info.publish(info)
                rclpy.spin_once(node, timeout_sec=0.25)
            if len(output) < (2 if tracking else 1):
                raise RuntimeError(f'{mode}: no localisation output')
            if tracking:
                assert output[0].object_ids == output[1].object_ids, 'persistent track IDs'
                assert len(output[0].object_ids) == len(output[0].objects)
            message = output[0]
            if not message.objects:
                raise RuntimeError(f'{mode}: no valid object geometry')
            signatures = []
            for obj in message.objects:
                # Transport correctness applies equally to legacy and hybrid.
                mask = obj.segmented_binary_mask
                assert mask.encoding == 'mono8', 'actual uint8 mask encoding'
                assert mask.step == mask.width, 'packed uint8 row stride'
                assert len(mask.data) == mask.step * mask.height, 'packed payload'
                assert mask.header == message.header, 'source mask header'
                assert obj.segmented_pcl.header == message.header, 'source cloud header'
                if mode in ('enabled', 'float', 'tracking'):
                    assert obj.centroid.z > 0.49
                signatures.append([obj.name, obj.roi.x_offset, obj.roi.y_offset,
                                   obj.roi.width, obj.roi.height,
                                   obj.centroid.x, obj.centroid.y, obj.centroid.z,
                                   obj.length, obj.breadth, obj.height])
            return signatures
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            node.destroy_subscription(sub)
            node.destroy_node()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('ROS_DOMAIN_ID'):
        raise RuntimeError('explicit isolated ROS_DOMAIN_ID required')
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    try:
        results = {mode: run(args.binary, args.image, mode, args.output)
                   for mode in ('default', 'disabled', 'enabled', 'float', 'tracking')}
        assert results['default'] == results['disabled'], 'disabled legacy parity'
        assert len(results['enabled']) == len(results['float']), 'encoding object count'
        for integer, floating in zip(results['enabled'], results['float']):
            assert integer[:5] == floating[:5]
            np.testing.assert_allclose(integer[5:], floating[5:], atol=1e-6, rtol=1e-6)
        (args.output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        print('PASS: default/disabled parity, enabled uint16/float geometry, '
              'ROS mask/cloud headers, persistent hybrid track IDs')
    finally:
        rclpy.shutdown()
