import time
import shutil
import logging
import multiprocessing as mp

from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.aggregate import aggregate_features_pool
from ad_nids.process.columns import UGR_COLUMNS, FLOW_COLUMNS, FLOW_STATS
from ad_nids.process.process_parser import get_argparser
from ad_nids.utils import yyyy_mm_dd2mmdd


DATASET_NAME = 'UGR_16'
FEATURES = 'BASIC'

# ToDo: change from mock dates to real
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


def create_aggr_ugr_dataset(aggr_path, train_dates, test_dates, frequency='T'):

    aggr_path = Path(aggr_path).resolve()

    dataset_name = '{}_TRAIN_{}_TEST_{}_{}_{}'.format(
        DATASET_NAME,
        '-'.join(yyyy_mm_dd2mmdd(train_dates)),
        '-'.join(yyyy_mm_dd2mmdd(test_dates)),
        frequency, FEATURES
    )

    meta = create_meta(DATASET_NAME, train_dates,
                       test_dates, frequency, FEATURES, name=dataset_name)
    logging.info("Creating dataset {}...".format(meta['name']))

    train_paths = []
    for dt in train_dates:
        train_paths.extend((aggr_path / str(dt)).glob(f'*aggr_{frequency}.csv'))

    test_paths = []
    for dt in test_dates:
        test_paths.extend((aggr_path / str(dt)).glob(f'*aggr_{frequency}.csv'))

    # Naive concat
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    train_meta = train.loc[:, ['sa', 'tws']]
    train = train.loc[:, FLOW_STATS.keys()]

    test = pd.concat([pd.read_csv(p) for p in test_paths])
    test_meta = test.loc[:, ['sa', 'tws']]
    test = test.loc[:, FLOW_STATS.keys()]

    logging.info("Done")

    return Dataset(train, test, train_meta, test_meta, meta)


def aggregate_ugr_data(split_root_path, aggr_path, processes=-1,
                       frequency='T', exist_ok=True):

    split_root_path = Path(split_root_path).resolve()
    aggr_path = Path(aggr_path).resolve()

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

        date_path = aggr_path / date
        date_path.mkdir(parents=True, exist_ok=True)

        for flow_path in flow_paths:

            path_basename = flow_path.name[:-len(flow_path.suffix)]
            out_name = "{}_aggr_{}.csv".format(path_basename, frequency)
            out_path = aggr_path / date / out_name

            # Don't overwrite if exist_ok is set
            if out_path.exists() and exist_ok:
                logging.info("Found existing; no overwrite")
                continue

            flows = pd.read_csv(flow_path)
            flows = format_ugr_flows(flows)

            aggr_flows = aggregate_features_pool(flows, frequency, processes)
            aggr_flows.to_csv(out_path, index=False)

        logging.info("Done {0:.2f}".format(time.time() - start_time))


if __name__ == '__main__':

    # Example command
    # python ugr16.py ../tests/data/ugr_mock_split/ ../tests/data/processed/ -p -1 -f T --overwrite --plot

    parser = get_argparser()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    train_dates = TRAIN_DATES
    test_dates = TEST_DATES

    aggr_dir = args.aggr_dir if args.aggr_dir else args.root_dir
    aggregate_ugr_data(args.root_dir, aggr_dir,
                       processes=args.processes, frequency=args.frequency)
    dataset = create_aggr_ugr_dataset(
        aggr_dir, train_dates=train_dates,
        test_dates=test_dates, frequency=args.frequency
    )

    dataset.write_to(args.out_dir, visualize=args.plot,
                     overwrite=args.overwrite, archive=args.archive)
