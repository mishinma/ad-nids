import argparse
import logging
import json
import shutil
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from alibi_detect.datasets import Bunch
from alibi_detect.utils.saving import load_detector, save_detector

import ad_nids.experiments.fit_predict as experiments
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_config
from ad_nids.utils.misc import set_seed

DEFAULT_CONTAM_PERCS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 12, 15, 20, 30, 40, 50, 70]
THRESHOLD_BATCH_N_SAMPLES = 10000
THRESHOLD_BATCH_PERC_OUTLIER = 5
PREPARE_DATA_RANDOM_SEED = 42


def parser_fit_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str,
                        help="log directory")
    parser.add_argument("--data_path", nargs='*',
                        help="data root path")
    parser.add_argument("--config_path", nargs='*',
                        help="directory with config files")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    parser.add_argument("--no-shuffle", action='store_true')
    return parser


def prepare_experiment_data(dataset_path):

    logging.info(f'Loading dataset {dataset_path.name}')
    dset = Dataset.from_path(dataset_path)

    X_train = dset.train.drop(columns=['target'])
    y_train = dset.train['target'].values
    X_threshold = dset.threshold.drop(columns=['target'])
    y_threshold = dset.threshold['target'].values
    X_test = dset.test.drop(columns=['target'])
    y_test = dset.test['target'].values

    X_train = dset.preprocessor.transform(X_train).astype(np.float32)
    X_threshold = dset.preprocessor.transform(X_threshold).astype(np.float32)
    X_test = dset.preprocessor.transform(X_test).astype(np.float32)

    train_normal_batch = Bunch(data=X_train,
                               target=y_train,
                               target_names=['normal', 'outlier'])
    threshold_batch = Bunch(data=X_threshold,
                            target=y_threshold,
                            target_names=['normal', 'outlier'])
    test_batch = Bunch(data=X_test,
                       target=y_test,
                       target_names=['normal', 'outlier'])

    return train_normal_batch, threshold_batch, test_batch


def runner_fit_predict():
    parser = parser_fit_predict()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    log_root = Path(args.log_path).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    shuffle_data = not args.no_shuffle

    config_paths = []
    for config_path in args.config_path:
        config_path = Path(config_path).resolve()
        if config_path.is_dir():
            config_paths += list(config_path.iterdir())
        else:
            config_paths.append(config_path)

    dataset_paths = []
    for data_path in args.data_path:
        data_path = Path(data_path).resolve()
        if Dataset.is_dataset(data_path):
            dataset_paths.append(data_path)
        else:
            dataset_paths += [p for p in data_path.iterdir()
                              if Dataset.is_dataset(p)]

    for dataset_path in dataset_paths:

        experiment_data = prepare_experiment_data(dataset_path)

        for config_path in config_paths:

            configs = pd.read_csv(config_path)
            configs['dataset_path'] = str(dataset_path)
            configs['dataset_name'] = dataset_path.name

            for idx, config in configs.iterrows():

                run_fn = config.get('run_fn', 'no_fn')
                try:
                    run_fn = getattr(experiments, run_fn)
                except AttributeError as e:
                    logging.error(f"No such function {run_fn}")
                    continue

                config = config.to_dict()
                num_tries = config.get('num_tries', 1)

                log_dir = get_log_dir(log_root, config)
                log_dir.mkdir()
                log_config(log_dir, config)

                # Create a directory to store experiment logs
                logging.info('Created a new log directory\n')
                logging.info(f'{log_dir}\n')

                for i_run in range(num_tries):

                    success = False

                    logging.info(f'Starting {config["config_name"]}')
                    logging.info(json.dumps(config, indent=2))
                    logging.info(f'RUN: {i_run}')
                    i_run_log_dir = log_dir / str(i_run)
                    i_run_log_dir.mkdir()

                    fh = logging.FileHandler(i_run_log_dir / 'run.log')
                    logger.addHandler(fh)

                    try:
                        # Pass data
                        run_fn(config, log_dir, experiment_data,
                               contam_percs=DEFAULT_CONTAM_PERCS, i_run=i_run, shuffle=shuffle_data)
                    except Exception as e:
                        logging.exception(e)
                    else:
                        success = True
                        logging.info('Successful')

                    if i_run == num_tries - 1:
                        logging.warning('Model did not converge!')

                    logger.removeHandler(fh)

                    if success:
                        break


def copy_log_path(log_path, dst_log_root):

    log_path = Path(log_path)
    dst_log_path = Path(dst_log_root)/log_path.name
    dst_log_path.mkdir(parents=True)

    with open(log_path / 'config.json', 'r') as f:
        config = json.load(f)

    log_config(dst_log_path, config)

    try:
        od = load_detector(str(log_path / 'detector'))
    except Exception as e:
        logging.exception('Detector does not exist')
    else:
        save_detector(od, str(dst_log_path / 'detector'))


def copy_log_paths(log_paths, dst_log_root):

    log_paths = [Path(p).resolve() for p in log_paths]
    for log_path in log_paths:
        copy_log_path(log_path, dst_log_root)


def runner_predict(log_paths, prepare_experiment_data_fn=prepare_experiment_data):
    loglevel = getattr(logging, 'INFO', None)
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # provided a single root_path
    if not isinstance(log_paths, list):
        log_paths = list(Path(log_paths).resolve().iterdir())
    log_paths = [Path(p).resolve() for p in log_paths]

    dataset2log_paths = {}
    for log_path in log_paths:

        with open(log_path / 'config.json', 'r') as f:
            config = json.load(f)

        dataset_path = Path(config['dataset_path']).resolve()
        dataset2log_paths.setdefault(dataset_path, []).append(log_path)

    for dataset_path, log_paths in dataset2log_paths.items():

        set_seed(PREPARE_DATA_RANDOM_SEED)
        experiment_data, _ = prepare_experiment_data_fn(dataset_path)

        for log_path in log_paths:

            with open(log_path / 'config.json', 'r') as f:
                config = json.load(f)

            logging.info(f'Starting {config["config_name"]}')
            logging.info(json.dumps(config, indent=2))

            run_fn = config.get('run_fn', 'no_fn')
            try:
                run_fn = getattr(experiments, run_fn)
            except AttributeError as e:
                logging.error(f"No such function {run_fn}")
                continue

            try:
                # Pass data
                run_fn(config, log_path, experiment_data,
                       contam_percs=DEFAULT_CONTAM_PERCS, load_outlier_detector=True)
            except Exception as e:
                logging.exception(e)
            else:
                logging.info('Successful')


if __name__ == '__main__':
    runner_fit_predict()
