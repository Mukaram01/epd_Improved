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
"""Small scoring regressions; no scene data, inference mocks or held-out inputs."""
import copy

import pytest

from compare_hybrid_rgbd import recovery_summary


def report(foreground, spill, contamination, published, valid):
    return {
        'result_sha256': 'same-sealed-export',
        'plan_sha256': 'same-plan', 'manifest_sha256': 'same-train-manifest',
        'totals': {'baseline': {'foreground_retention': 0.7},
                   'hybrid_candidate': {'foreground_retention': foreground / 100,
                                        'spill': spill, 'cross_object': contamination},
                   'hybrid_published': {'foreground_retention': published / 100}},
        'scenes': [{'scene': 'development', 'valid_3d': valid,
                    'rows': [{'owner': 5, 'prediction': 0,
                              'baseline': {'foreground': 70, 'spill': 30},
                              'hybrid_candidate': {'foreground': foreground, 'spill': spill,
                                                   'cross_object': contamination},
                              'hybrid_published': {'foreground': published}}]}]}


def test_reports_per_instance_spill_regression_despite_aggregate_acceptance():
    previous = report(60, 10, 3, 40, 1)
    current = report(69, 11, 4, 69, 2)
    summary = recovery_summary(previous, current)
    assert summary['development_targets_met']
    assert summary['per_instance'][0]['regressions'] == ['spill', 'cross_object']
    assert summary['per_instance'][0]['foreground_delta'] == 9
    assert summary['per_instance'][0]['published_foreground_delta'] == 29
    assert summary['background_only_spill'] == {'before': 7, 'after': 7}


def test_fails_development_targets_when_spill_or_contamination_cap_is_exceeded():
    previous = report(60, 10, 3, 40, 1)
    for spill, contamination in [(955, 0), (300, 256)]:
        current = report(69, spill, contamination, 69, 2)
        assert not recovery_summary(previous, current)['development_targets_met']


def test_refuses_different_evidence_or_changed_instance_assignment():
    previous = report(60, 10, 3, 40, 1)
    for field in ['result_sha256', 'plan_sha256', 'manifest_sha256', 'assignment']:
        current = copy.deepcopy(previous)
        if field == 'assignment':
            current['scenes'][0]['rows'][0]['prediction'] = 1
        else:
            current[field] = 'different'
        with pytest.raises(ValueError):
            recovery_summary(previous, current)
