import json
import os
import zipfile
import shutil

from datetime import datetime
from timeit import default_timer as timer
from uuid import uuid4

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from generative_nids.ml.models import ALGORITHM2WRAPPER


def load_data(config, data_root_dir):
    dataset_path = os.path.join(data_root_dir, config['data_hash'])

    if not os.path.exists(dataset_path):
        arc_path = dataset_path + '.zip'
        with zipfile.ZipFile(arc_path) as ziph:
            ziph.extractall(data_root_dir)

    train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

    with open(os.path.join(dataset_path, 'meta.json')) as f:
        dataset_meta = json.load(f)

    return train, test, dataset_meta


def get_log_dir(log_root_dir):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + str(uuid4())[:5]
    log_dir = os.path.join(log_root_dir, uniq_name)
    return log_dir


def create_model(config):
    return ALGORITHM2WRAPPER[config['algorithm']](config['model_params'])


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


def run(config, data_root_dir, log_root_dir):

    train, test, dataset_meta = load_data(config, data_root_dir)
    log_dir = get_log_dir(log_root_dir)

    # Preprocess data
    x_train_norm, x_train_anom, x_test, y_test = prepare_data(train, test, config)

    # Train and fit the model
    model = create_model(config)
    se = timer()
    model.fit(x_train_norm)
    time_fit = timer() - se

    # Compute anomaly scores
    se = timer()
    score_test = model.anomaly_score(x_test)
    time_test = timer() - se

    # Compute anomaly scores
    se = timer()
    score_train_norm = model.anomaly_score(x_train_norm)
    score_train_anom = model.anomaly_score(x_train_anom)
    time_train = timer() - se

    score_train = np.concatenate([score_train_norm, score_train_anom])
    y_train = np.concatenate([np.zeros_like(score_train_norm), np.ones_like(score_train_anom)])

    # Log everything
    os.makedirs(log_dir)
    try:
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        with open(os.path.join(log_dir, 'dataset_meta.json'), 'w') as f:
            json.dump(dataset_meta, f)

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
    data_root_dir = '../tests/data/processed'
    log_root_dir = '../tests/data/logs'

    config1 = {
        'algorithm': 'IsolationForest',
        'model_params': {'n_estimators': 100, 'behaviour': 'new', 'contamination': 'auto'},
        'data_hash': 'b98849baae8b39c7ca3ef19d375b278e',
        'data_standardization': False
    }

    config2 = {
        'algorithm': 'NearestNeighbors',
        'model_params': {'n_neighbors': 5, 'algorithm': 'kd_tree'},
        'data_hash': 'b98849baae8b39c7ca3ef19d375b278e',
        'data_standardization': True
    }

    run(config1, data_root_dir, log_root_dir)
    run(config2, data_root_dir, log_root_dir)