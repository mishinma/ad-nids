
import os
import json
import shutil
import hashlib
import multiprocessing as mp

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm


# Baseline Feature Set
# ToDo: move to another module and unit test
COMPUTE_FLOW_STATS = OrderedDict([
    ('num_f', lambda f: f.shape[0]),
    ('num_uniq_da', lambda f: f['da'].unique().shape[0]),
    ('num_uniq_dp', lambda f: f['dp'].unique().shape[0]),
    ('num_uniq_sp', lambda f: f['sp'].unique().shape[0]),
    ('entropy_da', lambda f: entropy(f['da'].value_counts())),
    ('entropy_dp', lambda f: entropy(f['dp'].value_counts())),
    ('entropy_sp', lambda f: entropy(f['sp'].value_counts())),
    ('avg_td', lambda f: np.mean(f['td'])),
    ('std_td', lambda f: np.std(f['td'])),
    ('avg_pkt', lambda f: np.mean(f['pkt'])),
    ('std_pkt', lambda f: np.std(f['pkt'])),
    ('avg_byt', lambda f: np.mean(f['byt'])),
    ('std_byt', lambda f: np.std(f['byt'])),
    ('lbl', lambda f: int(bool(np.sum(f['lbl']))))
])

FLOW_STATS_COLUMNS = ['sa', 'tws'] + list(COMPUTE_FLOW_STATS.keys())


def aggregate_extract_features(flows, freq='5T', processes=1):
    grouped = flows.groupby(['sa', pd.Grouper(key='ts', freq=freq)])
    pbar = tqdm(total=len(grouped))

    records = []

    def log_progress(result):
        records.append(result)
        pbar.update(1)

    if processes > 1:
        pool = mp.Pool(processes=processes)
        for name, grp in grouped:
            pool.apply_async(
                extract_features_wkr, ((name, grp),),
                callback=log_progress)
        pool.close()
        pool.join()
    else:
        for name, grp in grouped:
            res = extract_features_wkr((name, grp))
            log_progress(res)

    processed = pd.DataFrame.from_records(records, columns=FLOW_STATS_COLUMNS)
    return processed


def extract_features_wkr(arg):

    grp_name, flow_grp = arg

    flow_stats = []

    for f in COMPUTE_FLOW_STATS .values():
        flow_stats.append(f(flow_grp))

    record = (
        grp_name[0], grp_name[1], *flow_stats
    )

    return record


def create_meta(dataset_name, train_split, test_split, frequency,
                features, name=None, notes=None):

    if notes is None:
        notes = ''

    if name is None:
        name = '{}_TRAIN_{}_TEST_{}_{}_{}'.format(
            dataset_name, '-'.join(train_split), '-'.join(test_split), frequency, features
        )

    meta = {
        'data_hash': None,
        'dataset_name': dataset_name,
        'train_split': train_split,
        'test_split': test_split,
        'frequency': frequency,
        'features': features,
        'notes': notes,
        'name': name
    }

    return meta


def create_archive(root_path, train, test, meta):

    root_path = Path(root_path).resolve()
    dataset_path = root_path/meta['name']
    dataset_path.mkdir(parents=True)

    train_path = dataset_path/'train.csv'
    test_path = dataset_path/'test.csv'

    train.to_csv(train_path, index=None)
    test.to_csv(test_path, index=None)

    data_hash = compute_hash([train_path, test_path])

    meta['data_hash'] = data_hash

    with open(dataset_path/'meta.json', 'w') as f:
        json.dump(meta, f)

    shutil.make_archive(dataset_path, 'zip', root_path, meta['name'])
    shutil.rmtree(dataset_path)


def compute_hash(paths):
    """ Naive concat """

    # create hash
    data_hash = hashlib.md5()
    for path in paths:
        with open(path, 'rb') as f:
            data_hash.update(f.read())

    return data_hash.hexdigest()
