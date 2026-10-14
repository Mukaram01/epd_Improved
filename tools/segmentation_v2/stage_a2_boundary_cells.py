#!/usr/bin/env python3
"""One bounded read-only 2x2 diagnostic; no model loading or corrective output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import stage_a2_boundary_audit as a

RETAINED_REPORT_SHA = '4b5a82edf7169a64940f4b410e5680a2c4662dba7d29282006165c3b8a8630d9'
# Trusted committed evidence at base 5f7e325, independent of external archives.
MANIFEST_SHA = {
    '#221':'1fb3489aabcfdb55f76d54a13368e9d1cb58deae14af82c2ddee301f687d0f2a',
    '#222':'989107658eff3c95a5f0b73058dd824d2907dc59eb9042f6eccd007a9d9b3cf5',
}
SUPPORT = ('all_background', 'target_only', 'other_only', 'target_and_other')
CATEGORIES = tuple(f'{s}/{b}' for s in SUPPORT
                   for b in ('no_active_background', 'active_background'))
# Histogram uses float64 subtraction of the original float32 values. Bins are
# left-inclusive/right-exclusive, final bin right-inclusive (numpy contract).
MARGIN_EDGES = [-.5, -.25, -.1, -.05, 0., .05, .1, .25, .4, .5]


def margin_hist(values):
    return {'edges': MARGIN_EDGES,
            'counts': np.histogram(values.astype(np.float64) - .5,
                                   bins=MARGIN_EDGES)[0].tolist(),
            'intervals': '[left,right), final [left,right]'}


def cells(prob, box, identity, ids256, ids512):
    if (prob.shape != (512,512) or prob.dtype != np.float32 or
            not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any() or
            ids256.shape != (256,256) or ids512.shape != (512,512) or
            ids256.dtype != np.uint32 or ids512.dtype != np.uint32 or
            np.asarray(box).shape != (4,) or not np.isfinite(box).all()):
        raise ValueError('CELL_CONTRACT: corrupt probability, box or ID raster')
    if (type(identity) is not int or identity <= 0 or
            not (ids256 == identity).any() or not (ids512 == identity).any()):
        raise ValueError('CELL_IDENTITY: reference absent or invalid')
    native, mapped = a.project(prob, box)
    yx = np.argwhere(mapped & (ids256 == 0)).astype(np.int32)
    y = 2*yx[:,0,None] + np.array([0,0,1,1])
    x = 2*yx[:,1,None] + np.array([0,1,0,1])
    ids, probabilities, active = ids512[y,x], prob[y,x], native[y,x]
    x0,y0,x1,y1 = a.bounds(box)
    roi = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
    target = ids == identity
    other = (ids != 0) & ~target
    bg = ids == 0
    support = target.any(axis=1).astype(int) + 2*other.any(axis=1).astype(int)
    category = 2*support + (active & bg).any(axis=1).astype(int)
    k = active.sum(axis=1)
    counts = np.bincount(category, minlength=len(CATEGORIES))
    joint = np.zeros((len(CATEGORIES), 4), dtype=np.int64)
    for cat, occupancy in zip(category, k):
        joint[cat,occupancy-1] += 1
    summary = {
        'reference_identity': identity, 'cells': len(yx),
        'categories': dict(zip(CATEGORIES, counts.tolist())),
        'category_by_active_samples_1_to_4': dict(zip(CATEGORIES, joint.tolist())),
        'active_samples_histogram_1_to_4': [int((k == n).sum()) for n in range(1,5)],
        'native_id_samples': {'target': int(target.sum()), 'background': int(bg.sum()),
                              'other': int(other.sum())},
        'active_native_id_samples': {'target': int((active & target).sum()),
                                     'background': int((active & bg).sum()),
                                     'other': int((active & other).sum())},
        'union_inactive_slots': int((4-k).sum()),
        'all_probability_margin_histogram': margin_hist(probabilities.ravel()),
        'active_probability_margin_histogram': margin_hist(probabilities[active]),
        'cell_max_active_margin_histogram': margin_hist(
            np.where(active, probabilities, -np.inf).max(axis=1)),
        'category_max_active_margin_histograms': {
            label: margin_hist(np.where(active, probabilities, -np.inf).max(axis=1)[category == j])
            for j,label in enumerate(CATEGORIES)},
    }
    arrays = {'yx': yx, 'ids': ids, 'probabilities': probabilities, 'roi': roi,
              'active': active, 'category': category.astype(np.uint8)}
    return summary, arrays


def check_coordinates(recomputed, retained):
    if set(recomputed) != set(retained) or any(
            recomputed[k].dtype != retained[k].dtype or
            not np.array_equal(recomputed[k], retained[k]) for k in recomputed):
        raise ValueError('RETAINED_COORDINATES: recomputation differs')


def archive(path, expected):
    if not path.exists():
        return None
    if not expected or a.digest(path) != expected:
        raise ValueError('ARCHIVE_SHA: present archive unpinned or corrupt')
    return a.load_predictions(path)


def manifest_index(name, path):
    if a.digest(path) != MANIFEST_SHA[name]:
        raise ValueError('ARCHIVE_MANIFEST_SHA: changed trusted ' + name + ' manifest')
    rows = a.read_json(path)
    rows = rows['files'] if isinstance(rows,dict) else rows
    return {r['path']:r['sha256'] for r in rows}


def run(retained, stage, output):
    if output.exists() or output.is_symlink():
        raise ValueError('OUTPUT_NOT_EXCLUSIVE: preserve all previous evidence')
    report_path = retained / 'boundary_report.json'
    if a.digest(report_path) != RETAINED_REPORT_SHA:
        raise ValueError('RETAINED_REPORT_SHA: changed completed audit')
    prior = a.read_json(report_path)
    coords_path = retained / 'boundary_coordinates.npz'
    if a.digest(coords_path) != prior['diagnostic_coordinates_sha256']:
        raise ValueError('RETAINED_COORDINATES_SHA: changed completed audit')
    paths = {'stage_a2_result': stage / 'result.json',
             'stage_a2_process': stage / 'process.json',
             'stage_a2_predictions': stage / 'predictions.npz',
             'capture_manifest': stage / 'capture/raw_sha256.json'}
    for key,path in paths.items():
        if a.digest(path) != prior['inputs_sha256'][key]:
            raise ValueError('SOURCE_SHA: changed ' + key)
    recorded = a.read_json(paths['stage_a2_result'])
    pred = a.load_predictions(paths['stage_a2_predictions'])
    a.validate_recorded(stage, recorded, pred, expected_checkpoint_sha=a.EXPECTED_CANDIDATE_SHA)
    if [i for i,s in enumerate(pred['scores']) if s >= .8] != [0,1]:
        raise ValueError('CELL_IDENTITY: confident inventory drift')
    ids256,ids512,capture_hashes = a.read_capture(stage, recorded)
    summaries, arrays, recomputed = [], {}, {}
    for i,row in enumerate(prior['current']):
        if row['prediction_index'] != i or row['reference_identity'] != recorded['measurements'][i]['reference_identity']:
            raise ValueError('CELL_IDENTITY: recorded identity/order drift')
        identity = row['reference_identity']
        measured, coords = a.instance(pred, i, identity, ids256, ids512)
        if measured != row:
            raise ValueError('RETAINED_METRICS: current recomputation differs')
        recomputed.update(coords)
        summary, samples = cells(pred['masks'][i,0], pred['boxes'][i], identity, ids256, ids512)
        if summary['cells'] != row['mapped256_background']:
            raise ValueError('CELL_COUNTS: mapped count drift')
        summary['prediction_index'] = i
        summaries.append(summary)
        arrays.update({f'current_d{i}_{k}':v for k,v in samples.items()})
    with np.load(coords_path, allow_pickle=False) as bundle:
        check_coordinates(recomputed, {k:bundle[k] for k in bundle.files})
    inputs = {str(p):a.digest(p) for p in [report_path,coords_path,*paths.values(),
                                         Path(__file__),Path(a.__file__)]}
    previous = []
    for name,evidence,external in a.BASES:
        manifest_path = evidence / 'artifact_locations.json'
        index = manifest_index(name,manifest_path)
        path = external / 'predictions.npz'
        old_pred = archive(path,index.get(str(path)))
        # Reuse the original hash/capture/identity/metric consistency gates.
        old = a.baseline(name,evidence,external,ids256,ids512,recorded,pred)
        inputs[str(manifest_path)] = a.digest(manifest_path)
        inputs[str(evidence / 'stage_a2_result.json')] = a.digest(evidence / 'stage_a2_result.json')
        pinned_old = next(p for p in prior['previous'] if p['model'] == name)
        if old['result_sha256'] != pinned_old['result_sha256'] or old['summaries'] != pinned_old['summaries']:
            raise ValueError('ARCHIVE_IDENTITY: retained summary drift')
        old['boundary_cells'] = None
        if old_pred is not None:
            inputs[str(path)] = a.digest(path)
            old['boundary_cells'] = []
            old_recorded = a.read_json(evidence / 'stage_a2_result.json')
            for i in (0,1):
                identity = summaries[i]['reference_identity']
                if (a.bounds(old_pred['boxes'][i]) != tuple(old_recorded['association']['detections'][i]['bbox']) or
                        abs(float(old_pred['scores'][i]) - old_recorded['measurements'][i]['score']) > 1e-7):
                    raise ValueError('ARCHIVE_IDENTITY: box/score differs')
                s,v = cells(old_pred['masks'][i,0],old_pred['boxes'][i],identity,ids256,ids512)
                s['prediction_index'] = i
                old_native,old_map = a.project(old_pred['masks'][i,0],old_pred['boxes'][i])
                current_yx = arrays[f'current_d{i}_yx']
                shared = old_map[current_yx[:,0],current_yx[:,1]]
                cat = arrays[f'current_d{i}_category']
                s['current_cell_shared_categories'] = {label:int(((cat == j) & shared).sum()) for j,label in enumerate(CATEGORIES)}
                s['current_cell_added_categories'] = {label:int(((cat == j) & ~shared).sum()) for j,label in enumerate(CATEGORIES)}
                # Original old probabilities at every current false-background cell,
                # including old-negative cells; preserve old ROI separately.
                y = 2*current_yx[:,0,None] + np.array([0,0,1,1])
                x = 2*current_yx[:,1,None] + np.array([0,1,0,1])
                x0,y0,x1,y1 = a.bounds(old_pred['boxes'][i])
                arrays.update({f'{name[1:]}_d{i}_{k}':value for k,value in v.items()})
                arrays[f'{name[1:]}_d{i}_at_current_probabilities'] = old_pred['masks'][i,0][y,x]
                arrays[f'{name[1:]}_d{i}_at_current_roi'] = (y>=y0)&(y<y1)&(x>=x0)&(x<x1)
                arrays[f'{name[1:]}_d{i}_at_current_active'] = old_native[y,x]
                old['boundary_cells'].append(s)
        previous.append(old)
    result = {'schema':'epd_stage_a2_boundary_cells/v1',
              'decision':prior['decision'], 'strict_accepted':0, 'strict_total':2,
              'training_supervisor_evidence':prior['training_supervisor_evidence'],
              'supervisor_evidence_source':'Inherited from SHA-pinned completed audit; no provenance reconstructed',
              'inputs_sha256':inputs, 'capture_raw_sha256':capture_hashes,
              'checkpoint_sha256':prior['checkpoint_sha256'],
              'mask_threshold_strict_gt':.5, 'confidence_threshold':.8,
              'sample_order':['top_left','top_right','bottom_left','bottom_right'],
              'category_order':CATEGORIES,
              'category_definition':'Cross native target/other ID presence with any ROI-active native background sample. All-background/no-active-background is impossible for a selected cell.',
              'union_definition':'Any 1..4 ROI samples >0.5 selects the 256 cell. union_inactive_slots=sum(4-k) describes support expansion, not extra counted cells or causal error.',
              'limitations':prior['limitations'],
              'current':summaries,'previous':previous}
    # No source artifact writes; exclusive directory creation also guards races.
    output.mkdir(parents=True,exist_ok=False)
    sample_path = output / 'boundary_cells.npz'
    np.savez_compressed(sample_path,**arrays)
    result['samples_sha256'] = a.digest(sample_path)
    (output / 'boundary_cells_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained',type=Path,default=Path('/home/user/epd-v2-artifacts/mask-resolution-boundary-audit-001'))
    parser.add_argument('--stage-a2',type=Path,default=Path('/home/user/epd-v2-artifacts/mask-resolution-stage-a2-001'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        result = run(args.retained,args.stage_a2,args.output)
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)}),file=sys.stderr)
        return 2
    print(json.dumps({'decision':result['decision'],'current':result['current'],
                      'report':str(args.output / 'boundary_cells_report.json')},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
