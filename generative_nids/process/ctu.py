import os
import time
import logging
import multiprocessing as mp

from pathlib import Path

import numpy as np
import pandas as pd

from generative_nids.ml.dataset import Dataset, create_meta
from generative_nids.process.aggregate import aggregate_extract_features
from generative_nids.process.columns import CTU2FLOW_COLUMNS, FLOW_COLUMNS, FLOW_STATS
from generative_nids.process.argparser import get_argparser

DATASET_NAME = 'CTU_13'
FEATURES = 'BASIC'

ORIG_TRAIN_SCENARIOS = [3, 4, 5, 7, 10, 11, 12, 13]
ORIG_TEST_SCENARIOS = [1, 2, 6, 8, 9]
ALL_SCENARIOS = sorted(ORIG_TEST_SCENARIOS + ORIG_TRAIN_SCENARIOS)


def format_ctu_flows(flows):
    flows['StartTime'] = pd.to_datetime(flows['StartTime'], format='%Y/%m/%d %H:%M:%S.%f')
    flows = flows.sort_values(by='StartTime').reset_index(drop=True)
    flows = flows.rename(columns=CTU2FLOW_COLUMNS)[FLOW_COLUMNS]
    flows['lbl'] = flows['lbl'].str.contains('From-Botnet').astype(np.int)
    return flows


def process_ctu_data(root_dir, out_dir=None, processes=-1, frequency='T'):

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
        flows = format_ctu_flows(flows)

        aggr_flows = aggregate_extract_features(flows, frequency, processes)
        out_fname = "{}_aggr_{}.csv".format(flow_file, frequency)

        aggr_flows.to_csv(os.path.join(scenario_out_dir, out_fname), index=False)

        logging.info("Done {0:.2f}".format(time.time() - start_time))


def create_ctu_dataset(root_path, train_scenarios=None, test_scenarios=None, frequency='T'):

    root_path = Path(root_path).resolve()

    if train_scenarios is None:
        train_scenarios = ORIG_TRAIN_SCENARIOS

    if test_scenarios is None:
        test_scenarios = ORIG_TEST_SCENARIOS

    train_paths = []
    for sc in train_scenarios:
        train_paths.extend((root_path/str(sc)).glob(f'*aggr_{frequency}.csv'))

    test_paths = []
    for sc in test_scenarios:
        test_paths.extend((root_path / str(sc)).glob(f'*aggr_{frequency}.csv'))

    # Naive concat
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    train_meta = train.loc[:, ['sa', 'tws']]
    train = train.loc[:, FLOW_STATS.keys()]

    test = pd.concat([pd.read_csv(p) for p in test_paths])
    test_meta = test.loc[:, ['sa', 'tws']]
    test = test.loc[:, FLOW_STATS.keys()]

    meta = create_meta(DATASET_NAME, train_scenarios, test_scenarios, args.frequency, FEATURES)

    return Dataset(train, test, train_meta, test_meta, meta)


#ToDo change for real data
if __name__ == '__main__':

    parser = get_argparser()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    train_scenarios = ['2', '9']
    test_scenarios = ['3']

    # process_ctu_data(args.root_dir, args.out_dir, args.processes, args.frequency)
    dataset = create_ctu_dataset(
        args.root_dir, train_scenarios=train_scenarios,
        test_scenarios=test_scenarios, frequency=args.frequency
    )
    dataset.write_to('../tests/data/processed', plot=True)