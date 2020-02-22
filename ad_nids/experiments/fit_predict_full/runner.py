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

import ad_nids.experiments.fit_predict_full as experiments
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_config

DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]
THRESHOLD_BATCH_N_SAMPLES = 10000
THRESHOLD_BATCH_PERC_OUTLIER = 5


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
    return parser


def prepare_experiment_data(dataset):

    X_train = dataset.train.loc[dataset.train["target"] == 0]
    X_train = X_train.drop(columns=['target'])
    y_train = np.zeros((X_train.shape[0],), dtype=np.int)

    X_threshold, y_threshold = dataset.create_outlier_batch(train=True,
                                                                 n_samples=THRESHOLD_BATCH_N_SAMPLES,
                                                                 perc_outlier=THRESHOLD_BATCH_PERC_OUTLIER)
    X_test = dataset.test.drop(columns=['target'])
    y_test = dataset.test['target'].values

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


def runner_fit_predict():
    parser = parser_fit_predict()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
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

    for dataset_path in dataset_paths:

        logging.info(f'Loading dataset {dataset_path.name}')
        dataset = Dataset.from_path(dataset_path)
        experiment_data, preprocessor = prepare_experiment_data(dataset)

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
                logging.info(f'Starting {config["config_name"]}')
                logging.info(json.dumps(config, indent=2))

                num_tries = config.get('num_tries', 1)
                i_run = 0

                log_dir = get_log_dir(log_root, config)
                while True:
                    logging.info(f'Starting {config["config_name"]}')
                    logging.info(json.dumps(config, indent=2))
                    logging.info(f'RUN: {i_run}')
                    i_run += 1
                    log_dir.mkdir()
                    # Create a directory to store experiment logs
                    logging.info('Created a new log directory\n')
                    logging.info(f'{log_dir}\n')
                    logging.info(f'\n >>> tensorboard --host 0.0.0.0 --port 8888 --logdir {log_dir}\n')

                    try:
                        # Pass data
                        run_fn(config, log_dir, experiment_data, DEFAULT_CONTAM_PERCS)
                    except Exception as e:
                        logging.exception(e)
                        shutil.rmtree(str(log_dir))
                    else:
                        break
                    if i_run >= num_tries:
                        logging.warning('Model did NOT converge!')
                        break

                with open(log_dir / f'{i_run}.try', 'w') as f:
                    pass
                with open(log_dir / 'transformer.pickle', 'wb') as f:
                    pickle.dump(preprocessor, f)
                log_config(log_dir, config)


def runner_predict():
    pass


if __name__ == '__main__':
    runner_fit_predict()
