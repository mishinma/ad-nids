import shutil
import os
import logging

from datetime import datetime, time
from dateutil.parser import parse, parserinfo
from subprocess import CalledProcessError, check_output
from pathlib import Path

import pandas as pd
import numpy as np


from ad_nids.utils.exception import DownloadError
from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.misc import sample_df, dd_mm_yyyy2mmdd
from ad_nids.process.columns import CIC_IDS2017_COLUMN_MAPPING, CIC_IDS2017_ATTACK_LABELS


DATASET_NAME = 'CIC-IDS2017'
OFFICE_HOURS = (time(7, 0, 0), time(21, 0, 0))


def download_cicids17(data_path):

    logging.info('Downloading the dataset')

    data_path = Path(data_path).resolve()
    data_path.mkdir(parents=True)

    mycwd = os.getcwd()
    os.chdir(data_path)

    try:
        check_output(
            'wget --header="Host: 205.174.165.80" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" --header="Accept:'
            ' text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/'
            'signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" --header="Referer:'
            ' http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/" "http://205.174.165.80/CICDataset/CIC-IDS-2017/'
            'Dataset/GeneratedLabelledFlows.zip" -O "GeneratedLabelledFlows.zip" -c', shell=True
        )
    except Exception as e:
        raise DownloadError('Could not download the dataset')

    check_output(['unzip', "GeneratedLabelledFlows.zip"])
    os.remove(data_path / "GeneratedLabelledFlows.zip")

    middle_path = list(data_path.iterdir())[0]  # 'TrafficLabelling '
    for path in middle_path.iterdir():
        shutil.move(path, data_path / path.name)
    shutil.rmtree(middle_path)

    os.chdir(mycwd)  # go back where you came from

    # fix any possible enconding errors:
    for path in data_path.iterdir():
        with open(path, 'rb') as f:
            data = f.read()
        data = data.decode('utf-8', 'ignore')
        with open(path, 'w') as f:
            f.write(data)

    return


def cleanup_cidids17(data_path):

    logging.info('Cleaning up the data')

    data_path = Path(data_path).resolve()

    def parse_time(s):

        dt = parse(s, dayfirst=True)

        # they did not include am/pm info
        # if outside of office hours -> add pm
        if not OFFICE_HOURS[0] <= dt.time() <= OFFICE_HOURS[1]:
            dt = parse(s + ' pm', dayfirst=True)

        assert OFFICE_HOURS[0] <= dt.time() <= OFFICE_HOURS[1]

        return dt

    for path in data_path.iterdir():

        weekday = path.name.split('-')[0]
        assert parserinfo().weekday(weekday) is not None

        flows = pd.read_csv(path)
        flows = flows.dropna(how='all')
        flows = flows.rename(columns=lambda c: CIC_IDS2017_COLUMN_MAPPING.get(c.strip(), c))
        flows = flows[list(CIC_IDS2017_COLUMN_MAPPING.values())]
        assert len(flows.columns) == len(CIC_IDS2017_COLUMN_MAPPING)

        flows['timestamp'] = flows['timestamp'].apply(parse_time)
        day = flows['timestamp'].dt.strftime('%A')
        wrong_day_idx = day != weekday
        num_wrong_dates = np.sum(wrong_day_idx)
        if num_wrong_dates > 0:
            logging.info(f'Found {num_wrong_dates} flows with a wrong date')
            flows = flows[~wrong_day_idx]

        flows = flows.sort_values(by=['timestamp'])

        # Some rows in the files are just column names
        # Ignore them
        bad_rows = flows.index[flows['protocol'].astype(str).str.isalpha()]
        flows = flows.drop(bad_rows).reset_index(drop=True)

        # Fill na
        flows['flow_byts/s'] = flows['flow_byts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        flows['flow_pkts/s'] = flows['flow_pkts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # drop all other nan rows
        flows = flows.dropna(how='any')

        # Create a new file name
        attack_labels = list(flows['label'].unique())
        attack_labels.remove('BENIGN')
        attack_labels = list({CIC_IDS2017_ATTACK_LABELS[l] for l in attack_labels})
        if attack_labels:
            exp_labels = '-'.join(sorted(attack_labels))
        else:
            exp_labels = 'benign'

        exp_dt = flows.iloc[0]['timestamp']
        exp_date = exp_dt.strftime('%d-%m-%Y_%a')
        new_name = f'{exp_date}_{exp_labels}.csv'

        flows.to_csv(path, index=False)
        shutil.move(path, path.parent/new_name)

    return


if __name__ == '__main__':
    dataset_path = '/home/emikmis/data/cic-ids2017/data/CIC-IDS2017'
    cleanup_cidids17(dataset_path)
