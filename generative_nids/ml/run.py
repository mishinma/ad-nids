import json
import os
import shutil
import argparse
import logging

import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from generative_nids.ml.modelwrapper import create_model, is_param_required, FIT_PARAMS
from generative_nids.ml.dataset import Dataset
from generative_nids.ml.utils import precision_recall_curve_scores, select_threshold,\
    get_frontier, plot_precision_recall, plot_data_2d, plot_frontier

LOG_SHOW_PARAMS = ['dataset_name', 'algorithm', 'model_parameters',
                   'data_standardization', 'lr', 'num_epochs', 'optimizer']


def get_log_dir(config, log_root_dir):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + config['config_name']
    log_dir = Path(log_root_dir) / uniq_name
    log_dir = log_dir.resolve()
    return log_dir


def run(config, log_root_dir, frontier=False):

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
    model.fit(train_norm_loader.x, **fit_params)
    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')

    # Compute anomaly scores for train with anomalies
    # and select threshold
    se = timer()
    train_score = model.score(train_anom_loader.x)
    train_prf1_curve = precision_recall_curve_scores(train_anom_loader.y, train_score)
    model.threshold = select_threshold(train_prf1_curve['thresholds'],
                                       train_prf1_curve['f1scores'])
    y_train_pred = model.predict(train_score)
    train_cm = confusion_matrix(train_anom_loader.y, y_train_pred)
    train_prf1s = precision_recall_fscore_support(train_anom_loader.y,
                                                  y_train_pred, average='binary')
    time_train = timer() - se
    logging.info(f'Done (train): {time_train}')

    # Compute anomaly scores for test
    logging.info('Computing anomaly scores...')
    se = timer()
    test_score = model.score(test_loader.x)
    y_test_pred = model.predict(test_score)
    test_cm = confusion_matrix(test_loader.y, y_test_pred)
    test_prf1s = precision_recall_fscore_support(test_loader.y,
                                                 y_test_pred, average='binary')
    time_test = timer() - se
    logging.info(f'Done (test): {time_test}')

    # Log everything
    logging.info(f'Logging the results\n')
    log_dir.mkdir(parents=True)
    logging.info(f'{log_dir}\n')
    try:

        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        with open(os.path.join(log_dir, 'dataset_meta.json'), 'w') as f:
            json.dump(dataset.meta, f)

        model.save(log_dir)

        # plot train precision recall curve
        fig, ax = plt.subplots(1, 1)
        plot_precision_recall(
            ax,
            train_prf1_curve['precisions'], train_prf1_curve['recalls'], train_prf1_curve['thresholds']
        )
        fig.savefig(log_dir/'train_pr_curve.png')
        plt.close()

        # ToDo: move to another module
        if frontier:

            if test_loader.x.shape[1] == 2:

                fig, ax = plt.subplots(1, 1)
                xx, yy, Z = get_frontier(model, train_anom_loader.x)
                plot_frontier(ax, xx, yy, Z)
                plot_data_2d(ax, train_anom_loader.x_norm, train_anom_loader.x_anom)
                ax.set_title('Training frontier')
                fig.savefig(log_dir / 'train_frontier.png')
                plt.close()

                fig, ax = plt.subplots(1, 1)
                plot_frontier(ax, xx, yy, Z)
                plot_data_2d(ax, test_loader.x_norm, test_loader.x_anom)
                ax.set_title('Testing frontier')
                fig.savefig(log_dir / 'test_frontier.png')
                plt.close()

        eval_results = {
            'test_score': test_score.tolist(),
            'train_score': train_score.tolist(),
            'y_test_pred': y_test_pred.tolist(),
            'y_train_pred': y_train_pred.tolist(),
            'y_test': test_loader.y.tolist(),
            'y_train': train_anom_loader.y.tolist(),
            'threshold': model.threshold,
            'train_prf1_curve': train_prf1_curve,
            'train_prf1s': train_prf1s,
            'train_cm': train_cm.tolist(),
            'test_prf1s': test_prf1s,
            'test_cm': test_cm.tolist(),
            'time_train': time_train,
            'time_test': time_test,
            'time_fit': time_fit,
            'model_name': str(model)
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

