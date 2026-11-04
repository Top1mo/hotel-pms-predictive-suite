"""Orchestration: split first; fit transformations only on training rows."""
from pathlib import Path
import hashlib
import json
import platform
import pandas as pd
import sklearn
import yaml
from src.data.loader import load_h2, CATEGORICAL_COLUMNS
from src.data.splits import split_bookings
from src.preprocessing.cleaning import clean_dataset, cleaning_report
from src.preprocessing.pipelines import make_preprocessor, task_data
ROOT = Path(__file__).resolve().parents[2]

def prepare(config_path=None):
    config_path = Path(config_path).resolve() if config_path else ROOT/'config.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    raw_path = Path(cfg['data']['raw_path'])
    if not raw_path.is_absolute():
        raw_path = ROOT/raw_path
    raw = load_h2(raw_path, cfg['data']['expected_rows'])
    clean = clean_dataset(raw)
    p = cfg['project']
    parts, manifest = split_bookings(clean, p['random_seed'], p['test_size'], p['val_size'])
    out = ROOT/'data'/'splits'
    out.mkdir(parents=True, exist_ok=True)
    for name,part in parts.items():
        part.to_csv(out/f'{name}.csv', index=True)
    manifest.to_csv(out/'manifest.csv', index=False)
    report = cleaning_report(raw, clean)
    report.update({'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'seed':p['random_seed'], 'strategy':'grouped random split on booking-time source inputs',
        'requested_fractions_of_groups':{'train':1-p['test_size']-p['val_size'], 'validation':p['val_size'], 'test':p['test_size']},
        'actual_rows':{k:len(v) for k,v in parts.items()},
        'actual_groups':manifest.groupby('split')['group_id'].nunique().to_dict(),
        'training_cancel_rate':float(parts['train']['IsCanceled'].mean()),
        'versions':{'python':platform.python_version(), 'pandas':pd.__version__, 'sklearn':sklearn.__version__}})
    reports = ROOT/'outputs'/'person3'
    reports.mkdir(parents=True, exist_ok=True)
    (reports/'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return raw, parts, report

def load_partition(name):
    if name not in ('train','validation','test'):
        raise ValueError(name)
    return pd.read_csv(ROOT/'data'/'splits'/f'{name}.csv', index_col='row_id', dtype={c:'string' for c in CATEGORICAL_COLUMNS})

def check_preprocessing(parts):
    import numpy as np
    from scipy import sparse
    results = {}
    for task in ('regression','classification','clustering'):
        X,y = task_data(parts['train'], task)
        V,_ = task_data(parts['validation'], task)
        pipeline = make_preprocessor(task)
        train = pipeline.fit_transform(X,y)
        val = pipeline.transform(V)
        for matrix in (train,val):
            values = matrix.data if sparse.issparse(matrix) else matrix
            if not np.isfinite(values).all():
                raise AssertionError(f'{task}: non-finite transformed values')
        assert train.shape[1] == val.shape[1]
        results[task] = {'training_shape':list(train.shape), 'validation_shape':list(val.shape), 'sparse':sparse.issparse(train)}
    (ROOT/'outputs'/'person3'/'preprocessing_checks.json').write_text(json.dumps(results,indent=2), encoding='utf-8')
    return results
