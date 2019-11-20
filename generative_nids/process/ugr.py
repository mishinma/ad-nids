
import os
import glob
import time
import shutil
import logging
import multiprocessing as mp

from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from generative_nids.process.aggregate import aggregate_extract_features, \
    compute_hash, create_archive
from generative_nids.process.columns import UGR_COLUMNS, FLOW_COLUMNS
from generative_nids.process.argparser import get_argparser

DATASET_NAME = 'UGR_16'

#ToDo: change from mock dates to real
TRAIN_DATES = ['2016-07-27', '2016-07-30']
TEST_DATES = ['2016-07-31']
ALL_DATES = TRAIN_DATES + TEST_DATES


def parse_hour_path(p):

    date_hour_str = '_'.join(str(p).split('/')[-2:])
    try:
        dt = datetime.strptime(date_hour_str, '%Y-%m-%d_%H.csv')
    except ValueError as e:
        return None

    return dt


def split_ugr_flows(all_flows_path, split_root_path):

    split_root_path = Path(split_root_path)
    all_flows = pd.read_csv(all_flows_path, header=None,
                            names=UGR_COLUMNS, chunksize=1000000)

    for i, chunk in enumerate(all_flows):
        chunk['te'] = pd.to_datetime(chunk['te'], format='%Y-%m-%d %H:%M:%S')

        for tstmp, grp in chunk.resample('H', on='te'):

            if grp.empty:
                continue

            grp_date = tstmp.strftime('%Y-%m-%d')
            grp_hour = tstmp.strftime('%H')

            grp_path = split_root_path / grp_date / grp_hour
            grp_path.mkdir(parents=True, exist_ok=True)

            grp_name = f'{grp.index[0]}_{grp.index[-1]}.csv'
            grp.to_csv(grp_path/grp_name, index=None)

    # Combine chunks
    date_hour2chunk_paths = defaultdict(list)

    for chunk_path in split_root_path.glob('**/*.csv'):
        date_hour = tuple(str(chunk_path).split('/')[-3: -1])
        date_hour2chunk_paths[date_hour].append(chunk_path)

    for date_hour, chunk_paths in date_hour2chunk_paths.items():
        date, hour = date_hour
        date_hour_flows_path = split_root_path / date / f'{hour}.csv'

        date_hour_flows = pd.concat(
            [pd.read_csv(chunk_path)
             for chunk_path in chunk_paths]
        )
        date_hour_flows = date_hour_flows.sort_values(by='te')

        try:
            date_hour_flows.to_csv(date_hour_flows_path, index=None)
        except Exception as e:
            raise e
        else:
            shutil.rmtree(split_root_path / date / hour)


def format_ugr_flows(flows):

    flows['te'] = pd.to_datetime(flows['te'], format='%Y/%m/%d %H:%M:%S')
    flows.sort_values(by='te').reset_index(drop=True)
    flows['ts'] = flows['te'] - pd.to_timedelta(flows['td'], unit='ms')
    flows['lbl'] = ~(flows['lbl'] == 'background')
    flows['lbl'] = flows['lbl'].astype(np.int)
    flows = flows[FLOW_COLUMNS]
    return flows


def create_train_test(root_dir,
                      train_dates=None,
                      test_dates=None,
                      frequency='T'):

    if train_dates is None:
        train_dates = TRAIN_DATES

    if test_dates is None:
        test_dates = TEST_DATES

    train_paths = glob.glob(
        os.path.abspath(f'{root_dir}/{train_dates}/*aggr_{frequency}.csv')
    )

    test_paths = glob.glob(
        os.path.abspath(f'{root_dir}/{test_dates}/*aggr_{frequency}.csv')
    )

    # ToDo: inefficient, reading twice
    # Compute hash
    data_hash = compute_hash(list(train_paths) + (test_paths))

    # Naive concat
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    test = pd.concat([pd.read_csv(p) for p in test_paths])

    return train, test, data_hash


def process_ugr_data(split_root_path, out_dir=None, processes=-1, frequency='T'):

    if out_dir is None:
        out_dir = split_root_path

    split_root_path = Path(split_root_path).resolve()
    out_dir = Path(out_dir).resolve()

    if processes == -1:
        processes = mp.cpu_count() - 1

    date2paths = defaultdict(list)
    for path in split_root_path.glob('**/*.csv'):
        dt = parse_hour_path(path)
        if dt is not None:
            date2paths[dt.strftime('%Y-%m-%d')].append(path)

    for date, flow_paths in date2paths.items():

        logging.info("Processing date {}...".format(date))
        start_time = time.time()

        for flow_path in flow_paths:

            flows = pd.read_csv(flow_path)
            flows = format_ugr_flows(flows)

            aggr_flows = aggregate_extract_features(flows, frequency, processes)
            path_basename = os.path.splitext(os.path.basename(flow_path))[0]
            out_name = "{}_aggr_{}.csv".format(path_basename, frequency)

            aggr_flows.to_csv(out_dir/date/out_name, index=False)

        logging.info("Done {0:.2f}".format(time.time() - start_time))


if __name__ == '__main__':

    parser = get_argparser()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    # root_path = Path('/home/emikmis/data/UGR16/')
    # flow_path = root_path/'uniq'/'july.week5.csv.uniqblacklistremoved'
    # new_root_path = Path('/home/emikmis/data/UGR16_Split/')
    # split_ugr_flows(flow_path, root_path)
    split_root_path = '../tests/data/ugr_mock_split'
    process_ugr_data(args.root_dir, processes=1, frequency=args.frequency)

    train, test, data_hash = create_train_test(
        args.root_dir, frequency=args.frequency
    )

    meta = {
        'data_hash': data_hash,
        'dataset': DATASET_NAME,
        'feature_set': 'basic'
    }

    create_archive('../tests/data/processed', train, test, data_hash, meta=meta)

