import shutil

from datetime import datetime
from subprocess import CalledProcessError, check_output
from pathlib import Path

import pandas as pd
import numpy as np

from ad_nids.utils.misc import sample_df
from ad_nids.process.columns import CIC_IDS_ORIG_COLUMNS


class DownloadError(Exception):
    pass


def download_cicids(dataset_path):

    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True)

    try:
        check_output(
            ['aws', 's3', 'sync',
             '--region', 'eu-north-1',
             '--no-sign-request', "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms",
             dataset_path / 'processed'])
        returncode = 0
    except CalledProcessError as e:
        returncode = e.returncode

    if returncode != 0:
        raise DownloadError('Could not download the dataset')

    return


def cleanup_cidids(dataset_path):

    flows_root_path = Path(dataset_path) / 'processed'

    # Fix a typo in the filename
    typo_paths = [p for p in flows_root_path.iterdir() if 'Thuesday' in p.name]
    for path in typo_paths:
        typo_name = path.name
        fixed_name = typo_name.replace('Thuesday', 'Tuesday')
        shutil.move(path, path.parent/fixed_name)

    # Rename original files
    for path in flows_root_path.iterdir():
        exp_day_date = path.name.split('_')[0]
        exp_datetime = datetime.strptime(exp_day_date, '%A-%d-%m-%Y')
        new_name = exp_datetime.strftime('%d-%m-%Y-%a.csv').lower()
        shutil.move(path, path.parent/new_name)

    def format_col(c):
        return c.lower().replace(' ', '_').replace('/', '_')

    # Some rows in the files are just column names
    # Ignore them
    for path in flows_root_path.iterdir():

        flows = pd.read_csv(path)
        bad_rows = list(
            flows[flows['Protocol'] == 'Protocol'].index
        )
        flows = flows.drop(bad_rows).reset_index()

        flows = flows[CIC_IDS_ORIG_COLUMNS]
        flows = flows.rename({c: format_col(c) for c in flows.columns}, axis=1)

        # Fill na
        flows['flow_byts_s'] = flows['flow_byts_s'].fillna(0.0).astype(np.float64)
        flows['flow_pkts_s'] = flows['flow_pkts_s'].astype(np.float64)

        assert not flows.isnull().values.any()
        flows.to_csv(path, index=False)


def create_mock_cicids(dataset_path, mock_dataset_path, mock_dates=None,
                       num_normal_sample=10000, num_attack_sample=5000):

    if mock_dates is None:
        mock_dates = ['21-02-2018', '22-02-2018']

    dataset_path = Path(dataset_path)
    mock_dataset_path = Path(mock_dataset_path)

    processed_path = dataset_path / 'processed'
    mock_processed_path = mock_dataset_path / 'processed'

    mock_dataset_path.mkdir(parents=True)

    paths = [p for p in processed_path.iterdir()
             if '-'.join(p.name.split('-')[:3]) in mock_dates]

    for path in paths:

        flows = pd.read_csv(path)
        attack_idx = flows['label'] != 'Benign'
        flows_attack, flows_normal = flows[attack_idx], flows[~attack_idx]
        flows_attack_sample = sample_df(flows_attack, num_attack_sample)
        flows_normal_sample = sample_df(flows_normal, num_normal_sample)
        flows_sample = pd.concat([flows_attack_sample, flows_normal_sample])
        flows_sample = flows_sample.sort_values('Timestamp')
        flows_sample.to_csv(mock_processed_path/path.name, index=False)
