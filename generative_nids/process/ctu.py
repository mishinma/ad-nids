import argparse
import os
import glob
import time
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd

from generative_nids.process.aggregate import aggregate_extract_features, FLOW_COLUMNS

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


def format_flows(flows):
    formtd = flows.rename(columns=CTU2FLOW_COLUMNS)[FLOW_COLUMNS]
    formtd['lbl'] = formtd['lbl'].str.contains('From-Botnet').astype(np.int)
    return formtd


def process_dataset(root_dir, out_dir=None, processes=-1, frequency='T'):

    if out_dir is None:
        out_dir = root_dir

    if processes == -1:
        processes = mp.cpu_count() - 1

    scenarios = os.listdir(root_dir)
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

    return train, test


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

    process_dataset(args.root_dir, args.out_dir, args.processes, args.frequency)
