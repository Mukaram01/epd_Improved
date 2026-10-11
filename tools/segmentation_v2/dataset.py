#!/usr/bin/env python3
"""Metadata-first300/12/12 loader. Original pilot directories remain supported."""
import json
from pathlib import Path
import numpy as np
from pilot import sha, validate_native
from train import verified_rows as pilot_rows


def enumerate_records(manifest,split):
    if split not in ('train','validation','test'):raise ValueError('unknown split')
    data=json.loads(Path(manifest).read_text())
    if data['schema']!='epd_v2_dataset/2':raise ValueError('dataset schema')
    records=data['records']
    for field in ('scene_id','seed'):
        if len({r[field] for r in records})!=len(records):raise ValueError('duplicate dataset ID/seed')
    if len(records)!=324:raise ValueError('dataset record count')
    if {s:sum(r['split']==s for r in records) for s in ('train','validation','test')}!={'train':300,'validation':12,'test':12}:
        raise ValueError('dataset membership')
    from dataset_bindings import validate_metadata
    validate_metadata(data)
    return [r for r in records if r['split']==split]


def verified_rows(source,split):
    source=Path(source)
    if source.is_dir():
        yield from pilot_rows(source,split);return
    # Filter all memberships before any per-scene filesystem access.
    for row in enumerate_records(source,split):
        folder=Path(row['folder'])
        for name,digest in row['sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('dataset artifact changed')
        report=json.loads((folder/'capture.json').read_text())
        ids=np.fromfile(folder/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
        validate_native(report,ids)
        yield row,folder,report,ids
