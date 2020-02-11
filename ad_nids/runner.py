import argparse
import logging
import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from alibi_detect.datasets import Bunch

import ad_nids
import ad_nids.experiments as experiments
from ad_nids.dataset import Dataset
from ad_nids.report import create_experiments_report, create_datasets_report
from ad_nids.utils.logging import get_log_dir, log_config

DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]


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
    parser.add_argument("--report_path", type=str, default=None,
                        help="report directory")
    parser.add_argument("--contam_percs", default=None)
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser


def prepare_experiment_data(dataset_path):
    logging.info('Loading the dataset...')
    dataset = Dataset.from_path(dataset_path)

    train_targets = dataset.train.iloc[:, -1].values
    train_data_normal = dataset.train.iloc[train_targets == 0, :-1].values.astype(np.float32)
    train_data_outlier = dataset.train.iloc[train_targets == 1, :-1].values.astype(np.float32)

    train_data_normal, val_data = train_test_split(train_data_normal, test_size=0.1)

    train_targets_normal = np.zeros((train_data_normal.shape[0],), dtype=train_targets.dtype)
    train_targets_outlier = np.ones((train_data_outlier.shape[0],), dtype=train_targets.dtype)
    val_targets = np.zeros((val_data.shape[0],), dtype=train_targets.dtype)

    test_targets = dataset.test.iloc[:, -1].values
    test_data = dataset.test.iloc[:, :-1].values.astype(np.float32)

    train_normal_batch = Bunch(data=train_data_normal,
                               target=train_targets_normal,
                               target_names=['normal', 'outlier'])
    train_outlier_batch = Bunch(data=train_data_outlier,
                                target=train_targets_outlier,
                                target_names=['normal', 'outlier'])
    val_batch = Bunch(data=val_data,
                      target=val_targets,
                      target_names=['normal', 'outlier'])
    test_batch = Bunch(data=test_data,
                       target=test_targets,
                       target_names=['normal', 'outlier'])

    return train_normal_batch, train_outlier_batch, val_batch, test_batch


def runner_fit_predict():
    parser = parser_fit_predict()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    log_root = Path(args.log_path).resolve()

    if args.contam_percs is not None:
        contam_percs = json.loads(args.contam_percs)
    else:
        contam_percs = DEFAULT_CONTAM_PERCS

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
        experiment_data = prepare_experiment_data(dataset_path)
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
                run_fn(config, log_path, experiment_data, do_plot_frontier=True,
                       contam_percs=contam_percs)
            except Exception as e:
                logging.exception(e)


def runner_predict():
    pass


if __name__ == '__main__':
    runner_fit_predict()
