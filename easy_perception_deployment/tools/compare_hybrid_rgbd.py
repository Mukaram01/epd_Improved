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

"""One fixed comparison of the production C++ helper on two sealed train scenes.

Ownership labels enter scoring only. No model, held-out payload or pose is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

SCENES = ('expansion_093', 'expansion_122')
MANIFEST_SHA = '04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11'
RESULT_SHA = '5a471dfe7a93048b17463c91fd287a96e351fae1c4eda3deb455b443bb6d8497'
PLAN_SHA = '807215a9a9d9c417a043d85936eea3ade605d6a6cc1d897a117a7d2473fde004'
MODEL_SHA = '21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(bundle, manifest_path, replay):
    # Pinned metadata and full seals are checked before loading scene arrays.
    for path, sha in ((bundle / 'result.json', RESULT_SHA),
                      (bundle / 'plan.json', PLAN_SHA), (manifest_path, MANIFEST_SHA)):
        if digest(path) != sha:
            raise ValueError('unrecognized development provenance')
    result = json.loads((bundle / 'result.json').read_text())
    plan = json.loads((bundle / 'plan.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    if (tuple(plan['scenes']) != SCENES or plan['model_sha256'] != MODEL_SHA or
            tuple(row['scene_id'] for row in result['scenes']) != SCENES or
            result['model_sha256_after'] != MODEL_SHA or
            (bundle / 'INVALID_CALIBRATION.json').exists()):
        raise ValueError('invalid export')
    for row in result['scenes']:
        source = next(r for r in manifest['records'] if r['scene_id'] == row['scene_id'])
        if source['split'] != 'train':
            raise ValueError('train-only comparison')
        for name, sha in row['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe seal path')
            if digest(bundle / row['scene_id'] / name) != sha:
                raise ValueError('scene seal mismatch')
    report = {'scope': 'two frozen synthetic training scenes; diagnostic only',
              'baseline': 'frozen original ONNX consumer masks; not legacy P3 pose parity',
              'result_sha256': RESULT_SHA, 'plan_sha256': PLAN_SHA,
              'manifest_sha256': MANIFEST_SHA, 'replay_sha256': digest(replay),
              'parameters': {'neighbor_m': 0.015, 'component_fraction': 0.60,
                             'overlap_fraction': 0.25, 'max_depth_m': 2.0,
                             'size_check': 'off'},
              'held_out_payloads_opened': 0, 'stage_a2': 'BLOCKED 0/2', 'scenes': []}
    for scene in SCENES:
        folder = bundle / scene
        with np.load(folder / 'predictions.npz', allow_pickle=False) as arrays:
            masks = arrays['consumer_masks'].astype(bool)
        if masks.shape[1:] != (512, 512) or len(masks) > 12:
            raise ValueError('bounded shape/count')
        camera = json.loads((folder / 'prediction_summary.json').read_text())['camera']
        with tempfile.TemporaryDirectory(prefix='epd-hybrid-') as temp:
            temp = Path(temp)
            masks.astype('u1').tofile(temp / 'input.u8')
            start = time.perf_counter()
            subprocess.run([str(replay), str(folder / 'capture.json.native512.depth_m.f32'),
                            str(temp / 'input.u8'), str(temp / 'output'), str(len(masks)),
                            *[str(camera[k]) for k in ('fx', 'fy', 'ppx', 'ppy')], '2.0'],
                           check=True, timeout=30, capture_output=True)
            elapsed = time.perf_counter() - start
            candidates = np.fromfile(temp / 'output.masks.u8', 'u1').reshape(masks.shape) > 0
            geometry = json.loads((temp / 'output.json').read_text())
        # Labels are first loaded AFTER inference; never passed to the executable.
        labels = np.fromfile(folder / 'capture.json.native512.ids.u32', '<u4').reshape(512, 512)
        depth = np.fromfile(folder / 'capture.json.native512.depth_m.f32', '<f4').reshape(512, 512)
        capture = json.loads((folder / 'capture.json').read_text())
        owners = sorted(m['renderer_id'] for m in capture['mapping'] if m['class_id'] == 1)
        edges = []
        for i, mask in enumerate(masks):
            for owner in owners:
                truth = labels == owner
                union = np.count_nonzero(mask | truth)
                iou = np.count_nonzero(mask & truth) / union if union else 0
                edges.append((iou, i, owner))
        assigned, used = {}, set()
        for iou, i, owner in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
            if iou > 0 and owner not in assigned and i not in used:
                assigned[owner] = i
                used.add(i)
        rows = []
        for owner in owners:
            i = assigned.get(owner)
            truth = labels == owner
            row = {'owner': owner, 'prediction': i, 'gt_pixels': int(truth.sum())}
            for arm, values in (('baseline', masks), ('hybrid_candidate', candidates),
                                ('hybrid_published', candidates)):
                mask = np.zeros((512, 512), bool) if i is None else values[i]
                if arm == 'hybrid_published' and i is not None and not geometry[i]['valid']:
                    mask = np.zeros((512, 512), bool)
                foreground = int(np.count_nonzero(mask & truth))
                spill = int(np.count_nonzero(mask & ~truth))
                cross = int(np.count_nonzero(mask & np.isin(labels, owners) & ~truth))
                row[arm] = {'foreground': foreground, 'spill': spill, 'cross_object': cross,
                            'missing': int(truth.sum()) - foreground,
                            'iou': foreground / (int(truth.sum()) + spill)}
            # Visible-surface references use labelled measured depth for scoring only.

            def measured_centroid(mask):
                y, x = np.where(mask & np.isfinite(depth) & (depth > 0))
                if not len(x):
                    return None
                z = depth[y, x].astype(float)
                return np.array([np.mean((x-camera['ppx'])/camera['fx']*z),
                                 np.mean((y-camera['ppy'])/camera['fy']*z), np.mean(z)])
            reference = measured_centroid(truth)
            baseline_center = None if i is None else measured_centroid(masks[i])
            row['baseline_surface_centroid_error_m'] = (
                None if baseline_center is None else
                float(np.linalg.norm(baseline_center-reference)))
            row['hybrid_surface_centroid_error_m'] = (
                None if i is None or not geometry[i]['valid'] else
                float(np.linalg.norm(np.array(geometry[i]['centroid_m'])-reference)))
            rows.append(row)
        unmatched = {}
        for arm, values in (('baseline', masks), ('hybrid_candidate', candidates),
                            ('hybrid_published', candidates)):
            extra_masks = [values[i] for i in range(len(masks)) if i not in used and
                           (arm != 'hybrid_published' or geometry[i]['valid'])]
            unmatched[arm] = {
                'false_positive_pixels': sum(int(mask.sum()) for mask in extra_masks),
                'cross_object_pixels': sum(int(np.count_nonzero(mask & np.isin(labels, owners)))
                                           for mask in extra_masks)}
        report['scenes'].append({'scene': scene, 'geometry': geometry, 'rows': rows,
                                 'unmatched_support': unmatched,
                                 'unmatched_predictions': [
                                     i for i in range(len(masks)) if i not in used],
                                 'valid_3d': sum(g['valid'] for g in geometry),
                                 'rejected': sum(not g['valid'] for g in geometry),
                                 'refinement_process_seconds': elapsed})
    report['totals'] = {}
    for arm in ('baseline', 'hybrid_candidate', 'hybrid_published'):
        totals = {key: sum(row[arm][key] for scene in report['scenes'] for row in scene['rows'])
                  for key in ('foreground', 'spill', 'cross_object', 'missing')}
        totals['unmatched_false_positive_pixels'] = sum(
            scene['unmatched_support'][arm]['false_positive_pixels'] for scene in report['scenes'])
        totals['spill'] += totals['unmatched_false_positive_pixels']
        totals['cross_object'] += sum(
            scene['unmatched_support'][arm]['cross_object_pixels'] for scene in report['scenes'])
        totals['foreground_retention'] = (
            totals['foreground'] / (totals['foreground'] + totals['missing']))
        totals['owners_iou50'] = sum(
            row[arm]['iou'] >= .5 for scene in report['scenes'] for row in scene['rows'])
        report['totals'][arm] = totals
    return report


def recovery_summary(previous, current):
    """Development targets only; original Stage-A2 gates are not evaluated here."""
    for field in ('result_sha256', 'plan_sha256', 'manifest_sha256'):
        if previous[field] != current[field]:
            raise ValueError('different comparison provenance')

    def instances(report):
        return {(scene['scene'], row['owner']): row
                for scene in report['scenes'] for row in scene['rows']}
    before, after = instances(previous), instances(current)
    if before.keys() != after.keys():
        raise ValueError('different owner inventory')
    rows = []
    for key, old in before.items():
        new = after[key]
        if old['prediction'] != new['prediction']:
            raise ValueError('changed baseline assignment')
        a, b = old['hybrid_candidate'], new['hybrid_candidate']
        published_delta = (
            new['hybrid_published']['foreground'] - old['hybrid_published']['foreground'])
        regressions = []
        if b['foreground'] < a['foreground']:
            regressions.append('foreground')
        for metric in ('spill', 'cross_object'):
            if b[metric] > a[metric]:
                regressions.append(metric)
        if published_delta < 0:
            regressions.append('published_foreground')
        rows.append({'scene': key[0], 'owner': key[1], 'prediction': new['prediction'],
                     'original': old['baseline'], 'before': a, 'after': b,
                     'foreground_delta': b['foreground'] - a['foreground'],
                     'published_foreground_delta': published_delta, 'regressions': regressions})
    a, b = previous['totals']['hybrid_candidate'], current['totals']['hybrid_candidate']
    pub_a = previous['totals']['hybrid_published']['foreground_retention']
    pub_b = current['totals']['hybrid_published']['foreground_retention']
    valid_a = sum(scene['valid_3d'] for scene in previous['scenes'])
    valid_b = sum(scene['valid_3d'] for scene in current['scenes'])
    return {'development_targets_met': bool(
                b['foreground_retention'] > a['foreground_retention'] and
                b['spill'] <= 954 and b['cross_object'] <= 255 and
                pub_b > pub_a and valid_b > valid_a),
            'baseline_recovery_target_met': bool(
                b['foreground_retention'] >= current['totals']['baseline']['foreground_retention']),
            'candidate_foreground_retention': {'before': a['foreground_retention'],
                                               'after': b['foreground_retention']},
            'published_foreground_retention': {'before': pub_a, 'after': pub_b},
            'background_only_spill': {'before': a['spill'] - a['cross_object'],
                                      'after': b['spill'] - b['cross_object']},
            'geometry_accepted': {'before': valid_a, 'after': valid_b},
            'per_instance': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--reference', type=Path, help='Previous hybrid report on the same sealed evidence')
    args = parser.parse_args()
    comparison = compare(args.bundle, args.manifest, args.replay.resolve())
    if args.reference:
        comparison['recovery'] = recovery_summary(
            json.loads(args.reference.read_text()), comparison)
        comparison['reference_sha256'] = digest(args.reference)
    with args.output.open('x') as output:
        json.dump(comparison, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(comparison['totals'], indent=2))
