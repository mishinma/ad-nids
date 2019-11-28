import json
import os
import shutil
import argparse
import logging

from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler

from generative_nids.ml.modelwrapper import create_model, is_param_required, FIT_PARAMS
from generative_nids.ml.dataset import Dataset

LOG_SHOW_PARAMS = ['dataset_name', 'algorithm', 'model_parameters',
                   'data_standardization', 'lr', 'num_epochs', 'optimizer']


def get_log_dir(config, log_root_dir):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + config['config_name']
    log_dir = os.path.join(log_root_dir, uniq_name)
    return log_dir


def run(config, log_root_dir):

    logging.info(f'Starting {config["config_name"]}...')
    log_config = {k: v for k, v in config.items() if k in LOG_SHOW_PARAMS}
    logging.info(json.dumps(log_config, indent=2))

    # Create dataset and loaders
    logging.info('Loading the dataset...')
    dataset = Dataset.from_path(config['dataset_path'])

    if config['data_standardization']:
        x_train_norm = dataset.loader(train=True, contamination=False).x
        dataset.scaler = StandardScaler().fit(x_train_norm)

    log_dir = get_log_dir(config, log_root_dir)
    train_norm_loader = dataset.loader(train=True, contamination=False)
    train_anom_loader = dataset.loader(train=True, contamination=True)
    test_loader = dataset.loader(train=False, contamination=True)

    # Create a model
    algorithm = config['algorithm']
    model_params = config['model_parameters']
    if is_param_required('input_dim', algorithm):
        model_params['input_dim'] = test_loader.x.shape[1]
    model = create_model(algorithm, model_params)

    # Train the model on normal data
    logging.info('Training the model...')
    se = timer()
    fit_params = {k: v for k, v in config.items() if k in FIT_PARAMS}
    model.fit(train_norm_loader, **fit_params)

    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')

    # Compute anomaly scores for test
    logging.info('Computing anomaly scores...')
    se = timer()
    score_test = model.anomaly_score(test_loader)
    time_test = timer() - se
    logging.info(f'Done (test): {time_test}')

    # Compute anomaly scores for train with anomalies
    se = timer()
    score_train = model.anomaly_score(train_anom_loader)
    time_train = timer() - se
    logging.info(f'Done (train): {time_train}')

    # Log everything
    logging.info(f'Logging the results\n')
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
            'y_test': test_loader.y.tolist(),
            'y_train': train_anom_loader.y.tolist(),
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

    config_paths = sorted(config_paths)

    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = json.load(f)
        run(config, args.log_root_path)

