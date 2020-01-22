
import argparse
import logging
import json

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from alibi_detect.datasets import Bunch

import ad_nids.experiments as experiments
from ad_nids.dataset import Dataset
from ad_nids.report import create_experiments_report, create_datasets_report
from ad_nids.utils.logging import get_log_dir, log_config


DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]


def run_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_exp_path", type=str,
                        help="log directory")
    parser.add_argument("--config_path",  nargs='*',
                        help="directory with config files")
    parser.add_argument("--report_path", type=str, default=None,
                        help="report directory")
    parser.add_argument("--re", action="store_true",
                        help="re-evaluate existing log directories")
    parser.add_argument("--load", action="store_true",
                        help="load_outlier detector")
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


def runner():
    parser = run_parser()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    log_exp_path = Path(args.log_exp_path).resolve()

    if args.contam_percs is not None:
        contam_percs = json.loads(args.contam_percs)
    else:
        contam_percs = DEFAULT_CONTAM_PERCS

    if args.config_path:
        config_paths = []
        config_dset_paths = [Path(p).resolve() for p in args.config_path]
        for config_dset_path in config_dset_paths:
            config_paths.extend([p for p in config_dset_path.iterdir()
                                 if p.suffix == '.json'])

        log_paths = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            log_path = get_log_dir(log_exp_path, config["config_name"])
            log_path.mkdir(parents=True)
            log_config(log_path, config)
            log_paths.append(log_path)
    elif args.re:
        log_paths = list(log_exp_path.iterdir())
    else:
        raise ValueError('One of config_path or re must be provided')

    dataset_path2log_config = {}
    for log_path in log_paths:
        try:
            with open(log_path / 'config.json', 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.exception('Could not load config')
            continue

        log_configs = dataset_path2log_config.setdefault(config['dataset_path'], [])
        log_configs.append((log_path, config))

    for dataset_path, log_configs in dataset_path2log_config.items():

        experiment_data = prepare_experiment_data(dataset_path)

        for log_path, config in log_configs:

            try:
                run_fn = getattr(experiments, config['experiment_run_fn'])
            except AttributeError as e:
                logging.error(f"No such function "
                              f"{config['experiment_run_fn']}")
                continue

            try:
                # Pass data
                run_fn(config, log_path, experiment_data, do_plot_frontier=True,
                       contam_percs=contam_percs, load_outlier_detector=args.load)
            except Exception as e:
                logging.exception(e)


    if args.report_path is not None:

        report_path = Path(args.report_path).resolve()
        report_path.mkdir(parents=True)
        static_path = report_path / 'static'
        static_path.mkdir()

        log_paths = list([p for p in log_exp_path.iterdir() if p.is_dir()])
        log_paths = sorted(log_paths)

        datasets_report_path = report_path / 'datasets_report.html'
        logging.info(f"Creating all datasets report {datasets_report_path}")
        datasets_report = create_datasets_report(log_paths, static_path)
        with open(datasets_report_path, 'w') as f:
            f.write(datasets_report)

        experiments_report_path = report_path / 'experiments_report.html'
        logging.info(f"Creating all experiments report {experiments_report_path}")
        experiments_report = create_experiments_report(log_paths, static_path)
        with open(experiments_report_path, 'w') as f:
            f.write(experiments_report)


if __name__ == '__main__':
    runner()
