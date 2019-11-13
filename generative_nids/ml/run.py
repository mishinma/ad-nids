import json
import os
import zipfile

from datetime import datetime
from uuid import uuid4

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


def create_log_dir(log_root_dir):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + str(uuid4())[:5]
    log_dir = os.path.join(log_root_dir, uniq_name)
    os.makedirs(log_dir)
    return log_dir


def create_model(config):
    return ALGORITHM2WRAPPER[config['algorithm']](config['model_params'])


def prepare_data(data):
    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]
    return x, y


def run(config, data_root_dir, log_root_dir):

    train, test, dataset_meta = load_data(config, data_root_dir)
    log_dir = create_log_dir(log_root_dir)

    # dump config and dataset meta
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(log_dir, 'dataset_meta.json'), 'w') as f:
        json.dump(dataset_meta, f)

    x_train, y_train, x_test, y_test = prepare_data(train, test, config)

    # create a model
    model = create_model(config)

    model.fit(x_train)

    y_test_scores = model.anomaly_score(x_test)


if __name__ == '__main__':
    data_root_dir = '../tests/data/processed'
    log_root_dir = '../tests/data/logs'

    # config = {
    #     'algorithm': 'IsolationForest',
    #     'model_params': {'n_estimators': 100, 'behaviour': 'new', 'contamination': 'auto'},
    #     'data_hash': 'b98849baae8b39c7ca3ef19d375b278e',
    #     'data_standardization': False
    # }
    config = {
        'algorithm': 'NearestNeighbors',
        'model_params': {'n_neighbors': 5, 'algorithm': 'kd_tree'},
        'data_hash': 'b98849baae8b39c7ca3ef19d375b278e',
        'data_standardization': True,
    }

    run(config, data_root_dir, log_root_dir)