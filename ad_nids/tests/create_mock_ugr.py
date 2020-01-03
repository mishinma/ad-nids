import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from ad_nids.process.ugr import split_ugr_flows
from ad_nids.process.columns import UGR_COLUMNS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def is_in_daterange(timestamps, date_ranges):

    timestamps = pd.Series(timestamps)

    idx = pd.Series(np.zeros(timestamps.shape).astype(np.bool))

    for date_from, date_to in date_ranges:
        idx |= (timestamps >= date_from) & (timestamps < date_to)

    if idx.shape == (1,):
        idx = idx.iloc[0]

    return idx


def convert_dh_to_dt(date, hour):
    return datetime.strptime(f'{date} {hour}', '%Y-%m-%d %H')


def convert_ranges(date_ranges):
    date_ranges = [
        (convert_dh_to_dt(date, hours_range[0]),
         convert_dh_to_dt(date, hours_range[1]))
        for date, hours_range in date_ranges
    ]
    return date_ranges


def sample_normal_attack(attack_flows_path, all_flows_path,
                         date_ranges, chunksize, num_chunks_to_read):

    attack_flows = []

    # Flows per type of attack
    for flows_path in attack_flows_path.iterdir():

        try:
            flows = pd.read_csv(flows_path, header=None, names=FLOW_COLUMNS)
        except pd.errors.EmptyDataError:
            continue

        # filter flows by time
        flows['te'] = pd.to_datetime(flows['te'], format='%Y-%m-%d %H:%M:%S')
        flows = flows[is_in_daterange(flows['te'], date_ranges)]
        attack_flows.append(flows)

    attack_flows = pd.concat(attack_flows)

    # Now sample normal flows
    all_flows = pd.read_csv(all_flows_path, header=None,
                            names=UGR_COLUMNS, chunksize=chunksize)
    normal_flows = []

    for i, chunk in enumerate(all_flows):
        chunk_normal_flows = chunk[chunk['lbl'] == 'background']
        normal_flows.append(chunk_normal_flows)
        if i > num_chunks_to_read:
            break

    normal_flows = pd.concat(normal_flows)

    # NO SAMPLE :/
    # num_attack_sample = 10000
    # num_normal_sample = 15000
    # attack_flows = attack_flows.sample(num_attack_sample,
    #                                    replace=False, random_state=RANDOM_SEED)
    # normal_flows = normal_flows.sample(num_normal_sample,
    #                                    replace=False, random_state=RANDOM_SEED)

    normal_flows['te'] = pd.to_datetime(normal_flows['te'],
                                        format='%Y-%m-%d %H:%M:%S')
    flows = pd.concat([normal_flows, attack_flows])
    flows = flows.sort_values(by='te')

    return flows


if __name__ == '__main__':

    # path to downloaded week 5
    data_dir = Path(sys.argv[1])

    mock_dir = Path('data/mock_datasets/ugr_mock/')
    mock_split_dir = Path('data/mock_datasets/ugr_mock_split/')

    all_flows_path = data_dir / 'uniq' / 'july.week5.csv.uniqblacklistremoved'
    attack_flows_path = data_dir / 'july' / 'week5'

    mock_all_flows_path = mock_dir / 'july.week5.csv'

    if not mock_dir.exists():

        mock_dir.mkdir()

        date_ranges = convert_ranges([
            ('2016-07-30', ('04', '06')),
            ('2016-07-31', ('18', '20'))
        ])
        chunksize = 10000
        num_chunks_to_read = 10

        try:
            flows = sample_normal_attack(attack_flows_path, all_flows_path,
                                         date_ranges, chunksize, num_chunks_to_read)
            flows.to_csv(mock_all_flows_path, header=False, index=False)
        except Exception as e:
            mock_dir.rmdir()
            raise e

    if not mock_split_dir.exists():

        mock_split_dir.mkdir()
        try:
            split_ugr_flows(mock_all_flows_path, mock_split_dir)
        except Exception as e:
            mock_split_dir.rmdir()
            raise e
