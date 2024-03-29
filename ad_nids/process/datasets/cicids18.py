import shutil
import logging

from datetime import datetime
from subprocess import CalledProcessError, check_output
from pathlib import Path

import pandas as pd
import numpy as np

from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.exception import DownloadError
from ad_nids.utils.misc import sample_df, dd_mm_yyyy2mmdd
from ad_nids.process.columns import CIC_IDS2018_ATTACK_LABELS, CIC_IDS2018_COLUMN_MAPPING


DATASET_NAME = 'CSE-CIC-IDS2018'


def download_cicids18(data_path):

    logging.info('Downloading the dataset')

    data_path = Path(data_path)
    data_path.mkdir(parents=True)

    try:
        check_output(
            ['aws', 's3', 'sync',
             '--region', 'eu-north-1',
             '--no-sign-request', "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms",
             data_path])
        returncode = 0
    except CalledProcessError as e:
        returncode = e.returncode

    if returncode != 0:
        raise DownloadError('Could not download the dataset')

    # Fix a typo in the filename
    typo_paths = [p for p in data_path.iterdir() if 'Thuesday' in p.name]
    for path in typo_paths:
        typo_name = path.name
        fixed_name = typo_name.replace('Thuesday', 'Tuesday')
        shutil.move(path, path.parent / fixed_name)

    # Rename original files
    for path in data_path.iterdir():
        exp_day_date = path.name.split('_')[0]
        exp_datetime = datetime.strptime(exp_day_date, '%A-%d-%m-%Y')
        new_name = exp_datetime.strftime('%d-%m-%Y.csv').lower()
        shutil.move(path, path.parent / new_name)

    return


def cleanup_cidids18(data_path):

    logging.info('Cleaning up the data')

    for path in data_path.iterdir():

        flows = pd.read_csv(path)
        flows = flows.rename(columns=lambda c: CIC_IDS2018_COLUMN_MAPPING.get(c.strip(), c))
        flows = flows[list(CIC_IDS2018_COLUMN_MAPPING.values())]

        # ToDo: purge flows that are not from this date

        # Some rows in the files are just column names
        # Ignore them
        bad_rows = flows.index[flows['protocol'].astype(str).str.isalpha()]
        flows = flows.drop(bad_rows).reset_index(drop=True)

        # Fill na
        flows['flow_byts/s'] = flows['flow_byts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        flows['flow_pkts/s'] = flows['flow_pkts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)

        assert not flows.isnull().values.any()
        flows.to_csv(path, index=False)


def create_mock_cicids18(dataset_path, mock_dataset_path, mock_dates=None,
                       num_normal_sample=10000, num_attack_sample=5000):

    if mock_dates is None:
        mock_dates = ['21-02-2018', '22-02-2018']

    dataset_path = Path(dataset_path)
    mock_dataset_path = Path(mock_dataset_path)

    mock_dataset_path.mkdir(parents=True)

    paths = [p for p in dataset_path.iterdir()
             if p.name[:-len(p.suffix)] in mock_dates]

    for path in paths:

        flows = pd.read_csv(path)
        attack_idx = flows['label'] != 'Benign'
        flows_attack, flows_normal = flows[attack_idx], flows[~attack_idx]
        flows_attack_sample = sample_df(flows_attack, num_attack_sample)
        flows_normal_sample = sample_df(flows_normal, num_normal_sample)
        flows_sample = pd.concat([flows_attack_sample, flows_normal_sample])
        flows_sample = flows_sample.sort_values('timestamp')
        flows_sample.to_csv(mock_dataset_path/path.name, index=False)


def create_dataset_cicids18(dataset_path, train_dates, test_dates, create_hash=False):

    features = 'ORIG'

    dataset_name = '{}_TRAIN_{}_TEST_{}_{}'.format(
        DATASET_NAME,
        '-'.join(dd_mm_yyyy2mmdd(train_dates)),
        '-'.join(dd_mm_yyyy2mmdd(test_dates)),
        features
    )

    meta = create_meta(DATASET_NAME, train_dates, test_dates,
                       features=features, name=dataset_name)
    logging.info("Creating dataset {}...".format(meta['name']))

    train_paths = [dataset_path/f'{dt}.csv' for dt in train_dates]
    test_paths = [dataset_path/f'{dt}.csv' for dt in test_dates]

    # Naive concat
    # ToDo: protocol to meta
    meta_columns = ['protocol', 'timestamp', 'label']
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    train_meta = train.loc[:, meta_columns]
    train = train.drop(meta_columns, axis=1)
    train['target'] = 0
    train.loc[train_meta['label'].isin(CIC_IDS2018_ATTACK_LABELS), 'target'] = 1

    test = pd.concat([pd.read_csv(p) for p in test_paths])
    test_meta = test.loc[:, meta_columns]
    test = test.drop(meta_columns, axis=1)
    test['target'] = 0
    test.loc[test_meta['label'].isin(CIC_IDS2018_ATTACK_LABELS), 'target'] = 1

    logging.info("Done")

    return Dataset(train, test, train_meta, test_meta, meta, create_hash=create_hash)
