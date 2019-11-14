import argparse
import os
import json
import shutil
import hashlib
import glob
import time
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd

from generative_nids.process.aggregate import aggregate_extract_features, FLOW_COLUMNS

DATASET_NAME = 'ctu13'

FLOW2CTU_COLUMNS = {
    'ts': 'StartTime',
    'td': 'Dur',
    'sa': 'SrcAddr',
    'da': 'DstAddr',
    'sp': 'Sport',
    'dp': 'Dport',
    'pr': 'Proto',
    'pkt': 'TotPkts',
    'byt': 'TotBytes',
    'lbl': 'Label'
}
CTU2FLOW_COLUMNS = {v: k for k, v in FLOW2CTU_COLUMNS.items()}

ORIG_TRAIN_SCENARIOS = [3, 4, 5, 7, 10, 11, 12, 13]
ORIG_TEST_SCENARIOS = [1, 2, 6, 8, 9]
ALL_SCENARIOS = list(range(1, 14))


def format_flows(flows):
    formtd = flows.rename(columns=CTU2FLOW_COLUMNS)[FLOW_COLUMNS]
    formtd['lbl'] = formtd['lbl'].str.contains('From-Botnet').astype(np.int)
    return formtd


def process_dataset(root_dir, out_dir=None, processes=-1, frequency='T'):

    if out_dir is None:
        out_dir = root_dir

    if processes == -1:
        processes = mp.cpu_count() - 1

    scenarios = [s for s in os.listdir(root_dir) if s in map(str, ALL_SCENARIOS)]
    for scenario in scenarios:
        scenario_dir = os.path.join(root_dir, scenario)
        scenario_out_dir = os.path.join(out_dir, scenario)

        if not os.path.exists(scenario_out_dir):
            os.makedirs(scenario_out_dir)

        flow_file = [f for f in os.listdir(scenario_dir) if os.path.splitext(f)[1] == '.binetflow'][0]

        logging.info("Processing scenario {}...".format(scenario))
        start_time = time.time()

        flows = pd.read_csv(os.path.join(scenario_dir, flow_file))
        flows['StartTime'] = pd.to_datetime(flows['StartTime'], format='%Y/%m/%d %H:%M:%S.%f')
        flows = flows.sort_values(by='StartTime').reset_index(drop=True)
        flows = format_flows(flows)

        aggr_flows = aggregate_extract_features(flows, frequency, processes)
        out_fname = "{}_aggr_{}.csv".format(flow_file, frequency)

        aggr_flows.to_csv(os.path.join(scenario_out_dir, out_fname), index=False)

        logging.info("Done {0:.2f}".format(time.time() - start_time))


def create_train_test(root_dir,
                      train_scenarios=None,
                      test_scenarios=None,
                      frequency='T'):

    if train_scenarios is None:
        train_scenarios = ORIG_TRAIN_SCENARIOS

    if test_scenarios is None:
        test_scenarios = ORIG_TEST_SCENARIOS

    train_scenarios = list(map(int, train_scenarios))
    test_scenarios = list(map(int, test_scenarios))

    train_paths = glob.glob(
        os.path.abspath(f'{root_dir}/{train_scenarios}/*aggr_{frequency}.csv')
    )

    test_paths = glob.glob(
        os.path.abspath(f'{root_dir}/{test_scenarios}/*aggr_{frequency}.csv')
    )

    # Naive concat
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    test = pd.concat([pd.read_csv(p) for p in test_paths])

    # create hash
    data_hash = hashlib.md5()
    data_hash.update(train.to_csv(index=None).encode('utf-8'))
    data_hash.update(train.to_csv(index=None).encode('utf-8'))
    data_hash = data_hash.hexdigest()

    return train, test, data_hash


def create_archive(root_dir, train, test, data_hash, meta=None):

    root_dir = os.path.abspath(root_dir)
    dataset_dir = os.path.join(root_dir, data_hash)

    if os.path.exists(dataset_dir) or os.path.exists(f'{dataset_dir}.zip'):
        raise FileExistsError(f'Dataset {data_hash} exists!')
    os.makedirs(dataset_dir)

    train.to_csv(os.path.join(dataset_dir, 'train.csv'), index=None)
    test.to_csv(os.path.join(dataset_dir, 'test.csv'), index=None)
    if meta is None:
        meta = {}
    meta['data_hash'] = data_hash

    with open(os.path.join(dataset_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    shutil.make_archive(dataset_dir, 'zip', root_dir, data_hash)
    shutil.rmtree(dataset_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str,
                        help="dataset directory")
    parser.add_argument("-o", "--out_dir", type=str, default=None,
                        help="output directory")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of processes")
    parser.add_argument("-f", "--frequency", type=str, default='T',
                        help="time window scale")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    train_scenarios = ['2', '9']
    test_scenarios = ['3']

    # process_dataset(args.root_dir, args.out_dir, args.processes, args.frequency)
    train, test, data_hash = create_train_test(
        args.root_dir, train_scenarios=train_scenarios,
        test_scenarios=test_scenarios, frequency=args.frequency
    )

    meta = {
        'data_hash': data_hash,
        'dataset': DATASET_NAME,
        'feature_set': 'basic'
    }

    create_archive('../tests/data/processed', train, test, data_hash,
                   meta=meta)
