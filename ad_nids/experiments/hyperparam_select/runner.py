import argparse
import logging
import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path

import ad_nids
import ad_nids.experiments.fit_predict_full as experiments
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_config


DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]
DEFAULT_RANDOM_SEEDS = [11, 22, 33, 44, 55]
DEFAULT_SAMPLE_PARAMS = {
    'train': {'n_samples': 40000},
    'threshold': {'n_samples': 5000, 'perc_outlier': 5},
    'test': {'n_samples': 5000, 'perc_outlier': 5}
}


def parser_fit_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="directory with config files")
    parser.add_argument("log_path", type=str,
                        help="log directory")
    parser.add_argument("run_fn", type=str,
                        help="experiment run function")
    parser.add_argument("--data_path", nargs='*',
                        help="data root path")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser


def runner_fit_predict():
    parser = parser_fit_predict()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    log_root = Path(args.log_path).resolve()

    try:
        run_fn = getattr(experiments, args.run_fn)
    except AttributeError as e:
        logging.error(f"No such function {args.run_fn}")
        raise e

    dataset_paths = []
    for data_path in args.data_path:
        data_path = Path(data_path).resolve()
        if Dataset.is_dataset(data_path):
            dataset_paths += [data_path]
        else:
            dataset_paths += [p for p in data_path.iterdir()
                              if Dataset.is_dataset(p)]

    for dataset_path in dataset_paths:

        logging.info(f'Loading dataset {dataset_path.name}')
        dataset = Dataset.from_path(dataset_path)

        configs = pd.read_csv(args.config_path)
        configs['dataset_path'] = str(dataset_path)
        configs['dataset_name'] = dataset_path.name

        for idx, config in configs.iterrows():

            config = config.to_dict()
            log_path = get_log_dir(log_root, config)
            log_path.mkdir(parents=True)
            log_config(log_path, config)

            try:
                # Pass data
                run_fn(config, log_path, dataset,
                       DEFAULT_RANDOM_SEEDS, DEFAULT_SAMPLE_PARAMS, DEFAULT_CONTAM_PERCS)
            except Exception as e:
                logging.exception(e)


def runner_predict():
    pass


if __name__ == '__main__':
    runner_fit_predict()
