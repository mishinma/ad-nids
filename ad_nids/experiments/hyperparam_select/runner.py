import argparse
import logging
import json
import copy
import shutil
import pickle

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from alibi_detect.datasets import Bunch

import ad_nids.experiments.hyperparam_select as experiments
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_config
from ad_nids.utils.misc import set_seed, average_results, jsonify


DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]
DEFAULT_RANDOM_SEEDS = [11, 33, 55]
DEFAULT_SAMPLE_PARAMS = {
    'train': {'n_samples': 400000},
    'threshold': {'n_samples': 10000, 'perc_outlier': 5},
    'test': {'n_samples': 10000, 'perc_outlier': 5}
}


def parser_fit_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str,
                        help="log directory")
    parser.add_argument("--data_path", nargs='*',
                        help="data root path")
    parser.add_argument("--config_path", nargs='*',
                        help="directory with config files")
    parser.add_argument("--seeds", nargs='*',
                        help="random seeds")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser


def prepare_experiment_data(dataset, random_seed, sample_params):
    set_seed(random_seed)

    n_train_samples = sample_params['train']['n_samples']
    n_threshold_samples = sample_params['threshold']['n_samples']
    perc_threshold_outlier = sample_params['threshold']['perc_outlier']
    n_test_samples = sample_params['test']['n_samples']
    perc_test_outlier = sample_params['test']['perc_outlier']

    X_train, y_train = dataset.create_outlier_batch(train=True, n_samples=n_train_samples,
                                                    perc_outlier=0)
    X_threshold, y_threshold = dataset.create_outlier_batch(train=True, n_samples=n_threshold_samples,
                                                            perc_outlier=perc_threshold_outlier)
    X_test, y_test = dataset.create_outlier_batch(train=True, n_samples=n_test_samples,
                                                  perc_outlier=perc_test_outlier)

    numeric_features = dataset.meta['numerical_features']
    binary_features = dataset.meta['binary_features']
    categorical_feature_map = dataset.meta['categorical_feature_map']

    # normalize
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(categories=list(categorical_feature_map.values())),
         list(categorical_feature_map.keys())),
        ('bin', FunctionTransformer(), binary_features),
        ('num', StandardScaler(), numeric_features),
    ])

    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train).astype(np.float32)
    X_threshold = preprocessor.transform(X_threshold).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    train_normal_batch = Bunch(data=X_train,
                               target=y_train,
                               target_names=['normal', 'outlier'])
    threshold_batch = Bunch(data=X_threshold,
                            target=y_threshold,
                            target_names=['normal', 'outlier'])
    test_batch = Bunch(data=X_test,
                       target=y_test,
                       target_names=['normal', 'outlier'])

    return (train_normal_batch, threshold_batch, test_batch), preprocessor


def average_logs(log_root, log_root_ave):

    logging.info('Averaging the results in '.format(log_root))
    #  We average results and save in another log dir
    #  So that we can reuse report module for generating reports
    config2logs = {}

    for log_dir in log_root.iterdir():

        try:
            with open(log_dir/'config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            continue

        config_name = 'conf_' + config['config_name'].split('_')[1]  # exclude random seed
        config2logs.setdefault(config_name, []).append(log_dir)

    for config_name, log_dirs in config2logs.items():

        all_results = []
        for log_dir in log_dirs:
            try:
                with open(log_dir/'eval_results.json', 'r') as f:
                            results = json.load(f)
            except FileNotFoundError:
                continue
            all_results.append(results)

        log_ave_dir = log_root_ave / ('_'.join(log_dirs[0].name.split('_')[:-1] + ['AVE']))
        with open(log_dirs[0] / 'config.json', 'r') as f:
            config_ave = json.load(f)
        log_ave_dir.mkdir()
        ave_results = average_results(all_results)
        config_ave['config_name'] = config_name + '_AVE'
        log_config(log_ave_dir, config_ave)
        with open(log_ave_dir / 'eval_results.json', 'w') as f:
            json.dump(jsonify(ave_results), f)


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

    if args.seeds is None:
        random_seeds = DEFAULT_RANDOM_SEEDS
    else:
        random_seeds = [int(s) for s in args.seeds]

    for dataset_path in dataset_paths:

        logging.info(f'Loading dataset {dataset_path.name}')
        dataset = Dataset.from_path(dataset_path)

        for rs in random_seeds:

            experiment_data, preprocessor = prepare_experiment_data(dataset, rs, DEFAULT_SAMPLE_PARAMS)

            for config_path in config_paths:

                configs = pd.read_csv(config_path)
                configs['dataset_path'] = str(dataset_path)
                configs['dataset_name'] = dataset_path.name
                configs['random_seed'] = rs
                configs['config_name'] += '_{}'.format(rs)

                for idx, config in configs.iterrows():

                    try:
                        run_fn = getattr(experiments, config['run_fn'])
                    except AttributeError as e:
                        logging.error(f"No such function {run_fn}")
                        continue

                    config = config.to_dict()
                    logging.info(f'Starting {config["config_name"]}')
                    logging.info(json.dumps(config, indent=2))

                    log_dir = get_log_dir(log_root, config)

                    logging.info('Created a new log directory')
                    logging.info(f'{log_dir}\n')
                    log_dir.mkdir()
                    with open(log_dir / 'transformer.pickle', 'wb') as f:
                        pickle.dump(preprocessor, f)
                    log_config(log_dir, config)
                    num_tries = config.get('num_tries', 1)

                    for i_run in range(num_tries):

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
                                   contam_percs=DEFAULT_CONTAM_PERCS, i_run=i_run)
                        except Exception as e:
                            logging.exception(e)

                        if i_run == num_tries - 1:
                            logging.warning('Model did not converge!')

                        logger.removeHandler(fh)

    log_root_ave = log_root.parent/log_root.name + '_AVE'
    log_root_ave.mkdir(exist_ok=True)
    average_logs(log_root, log_root_ave)


def runner_predict():
    pass


if __name__ == '__main__':
    runner_fit_predict()
