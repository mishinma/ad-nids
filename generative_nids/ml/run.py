import json
import os
import shutil
import argparse
import logging

from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from sklearn.preprocessing import StandardScaler

from generative_nids.ml.modelwrapper import ALGORITHM2WRAPPER
from generative_nids.ml.dataset import Dataset


def get_log_dir(config, log_root_dir):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + config['config_name']
    log_dir = os.path.join(log_root_dir, uniq_name)
    return log_dir


def create_model(config):
    return ALGORITHM2WRAPPER[config['algorithm']](config['model_parameters'])


def prepare_data(train, test, config):
    x_train, y_train = train.iloc[:, 2:-1], train.iloc[:, -1].to_numpy()
    x_test, y_test = test.iloc[:, 2:-1], test.iloc[:, -1].to_numpy()

    train_idx_anom = y_train == 1
    x_train_norm = x_train[~train_idx_anom]
    x_train_anom = x_train[train_idx_anom]

    if config['data_standardization']:
        sc = StandardScaler()
        x_train_norm = sc.fit_transform(x_train_norm)
        x_train_anom = sc.transform(x_train_anom)
        x_test = sc.transform(x_test)

    return x_train_norm, x_train_anom, x_test, y_test


def run(config, log_root_dir):

    logging.info(f'Starting {config["config_name"]}...')

    logging.info('Loading the dataset...')
    dataset = Dataset.from_path(config['dataset_path'])

    log_dir = get_log_dir(config, log_root_dir)

    # Preprocess data
    x_train_norm, x_train_anom, x_test, y_test = prepare_data(dataset.train, dataset.test, config)

    logging.info('Training the model...')
    # Train and fit the model
    model = create_model(config)
    se = timer()
    model.fit(x_train_norm)
    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')

    logging.info('Computing anomaly scores...')
    # Compute anomaly scores
    se = timer()
    score_test = model.anomaly_score(x_test)
    time_test = timer() - se
    logging.info(f'Done (test): {time_test}')

    # Compute anomaly scores
    se = timer()
    score_train_norm = model.anomaly_score(x_train_norm)
    score_train_anom = model.anomaly_score(x_train_anom)
    time_train = timer() - se
    logging.info(f'Done (train): {time_test}')

    score_train = np.concatenate([score_train_norm, score_train_anom])
    y_train = np.concatenate([np.zeros_like(score_train_norm), np.ones_like(score_train_anom)])

    logging.info(f'Logging the results\n')
    # Log everything
    os.makedirs(log_dir)
    try:
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        with open(os.path.join(log_dir, 'dataset_meta.json'), 'w') as f:
            json.dump(dataset.meta, f)

        model.save(log_dir)

        eval_results = {
            'score_test': score_test.tolist(),
            'score_train': score_train.tolist(),
            'y_test': y_test.tolist(),
            'y_train': y_train.tolist(),
            'time_train': time_train,
            'time_test': time_test,
            'time_fit': time_fit
        }
        with open(os.path.join(log_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f)

    except Exception as e:
        shutil.rmtree(log_dir)
        raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="directory with config files")
    parser.add_argument("log_root_path", type=str,
                        help="log directory")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    config_root_path = Path(args.config_path).resolve()

    if config_root_path.is_dir():
        config_paths = list(config_root_path.iterdir())
    else:
        config_paths = [config_root_path]

    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = json.load(f)
        run(config, args.log_root_path)

