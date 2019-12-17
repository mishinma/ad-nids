import json
import os
import shutil
import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from alibi_detect.od import IForest
from alibi_detect.utils.saving import save_detector

from ad_nids.config import config_dumps
from ad_nids.dataset import Dataset
from ad_nids.report import create_experiments_report, create_datasets_report
from ad_nids.utils.logging import get_log_dir
from ad_nids.utils.metrics import precision_recall_curve_scores, get_frontier
from ad_nids.utils.plot import plot_precision_recall, plot_f1score, plot_data_2d, plot_frontier

EXPERIMENT_NAME = 'isolation_forest'
DEFAULT_CONTAM_PERCS = np.arange(1, 30, 5).tolist()


def run(config, log_exp_dir, do_plot_frontier=False):
    logging.info(f'Starting {config["config_name"]}')
    logging.info(config_dumps(config))

    if config["experiment_name"] != EXPERIMENT_NAME:
        logging.warning(
            'Experiment name mismatch. Expected {}, got {}.'.format(
                EXPERIMENT_NAME, config['experiment_name'])
        )

    # Create dataset and loaders
    logging.info('Loading the dataset...')
    dataset = Dataset.from_path(config['dataset_path'])

    normal_batch = dataset.create_outlier_batch(train=True, perc_outlier=0)
    X_train, y_train = normal_batch.data.astype('float'), normal_batch.target
    scaler = None
    if config['data_standardization']:
        scaler = StandardScaler().fit_transform(X_train)

    # Train the model on normal data
    logging.info('Fitting the model...')
    se = timer()
    od = IForest(threshold=None, n_estimators=config['n_estimators'])
    od.fit(X_train)
    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')

    log_dir = get_log_dir(log_exp_dir, config["config_name"])

    # Compute anomaly scores for train with anomalies
    # and select threshold
    threshold_batch = dataset.create_outlier_batch(train=True, scaler=scaler)
    X_threshold, y_threshold = threshold_batch.data.astype('float'), threshold_batch.target
    logging.info(f'Selecting the optimal threshold...')
    se = timer()
    od.infer_threshold(X_threshold, threshold_perc=100 - dataset.train_contamination_perc)
    X_threshold_pred = od.predict(X_threshold)
    y_threshold_pred = X_threshold_pred['data']['is_outlier']
    score_threshold = X_threshold_pred['data']['instance_score']
    time_score_train = timer() - se

    train_prf1_curve = precision_recall_curve_scores(
        y_threshold, score_threshold, 100 - DEFAULT_CONTAM_PERCS)
    train_cm = confusion_matrix(y_threshold, y_threshold_pred)
    train_prf1s = precision_recall_fscore_support(
        y_threshold, y_threshold_pred, average='binary')
    logging.info(f'Done (train): {timer() - se}')

    # Compute anomaly scores for test
    logging.info('Computing test anomaly scores...')
    test_batch = dataset.create_outlier_batch(train=False, scaler=scaler)
    X_test, y_test = test_batch.data.astype('float'), test_batch.target
    se = timer()
    X_test_pred = od.predict(X_test)
    y_test_pred = X_test_pred['data']['is_outlier']
    score_test = X_test_pred['data']['instance_score']
    time_score_test = timer() - se
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_prf1s = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    logging.info(f'Done (test): {timer() - se}')

    # Log everything
    logging.info(f'Logging the results\n')
    log_dir.mkdir(parents=True)
    od_save_path = log_dir / 'detector'
    od_save_path.mkdir()
    logging.info(f'{log_dir}\n')
    try:

        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        with open(os.path.join(log_dir, 'dataset_meta.json'), 'w') as f:
            json.dump(dataset.meta, f)

        save_detector(od, str(od_save_path))

        # plot train precision recall curve
        fig, ax = plt.subplots(1, 1)
        plot_precision_recall(
            ax,
            train_prf1_curve['precisions'], train_prf1_curve['recalls'], train_prf1_curve['threshold_percs']
        )
        fig.savefig(log_dir / 'train_pr_curve.png')
        plt.close()

        # plot train f1 score curve
        fig, ax = plt.subplots(1, 1)
        plot_f1score(
            ax,
            train_prf1_curve['f1scores'], train_prf1_curve['threshold_percs']
        )
        fig.savefig(log_dir / 'train_f1_curve.png')
        plt.close()

        # ToDo: move to another module
        if do_plot_frontier:

            if X_threshold.shape[1] == 2:
                fig, ax = plt.subplots(1, 1)
                xx, yy, Z = get_frontier(od, X_threshold)
                plot_frontier(ax, xx, yy, -Z)
                X_threshold_norm = X_threshold[~y_threshold.astype(bool)]
                X_threshold_anom = X_threshold[y_threshold.astype(bool)]
                plot_data_2d(ax, X_threshold_norm, X_threshold_anom)
                ax.set_title('Training frontier')
                fig.savefig(log_dir / 'train_frontier.png')
                plt.close()

                fig, ax = plt.subplots(1, 1)
                plot_frontier(ax, xx, yy, -Z)
                X_test_norm = X_test[~y_test.astype(bool)]
                X_test_anom = X_test[y_test.astype(bool)]
                plot_data_2d(ax, X_test_norm, X_test_anom)
                ax.set_title('Testing frontier')
                fig.savefig(log_dir / 'test_frontier.png')
                plt.close()

        eval_results = {
            'test_score': score_test.tolist(),
            'train_score': score_threshold.tolist(),
            'y_test_pred': y_test_pred.tolist(),
            'y_train_pred': y_threshold_pred.tolist(),
            'y_test': y_test.tolist(),
            'y_train': y_threshold.tolist(),
            'threshold': od.threshold,
            'train_prf1_curve': train_prf1_curve,
            'train_prf1s': train_prf1s,
            'train_cm': train_cm.tolist(),
            'test_prf1s': test_prf1s,
            'test_cm': test_cm.tolist(),
            'time_score_train': time_score_train,
            'time_score_test': time_score_test,
            'time_fit': time_fit,
            'model_name': od.meta['name']
        }

        with open(os.path.join(log_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f)

    except Exception as e:
        shutil.rmtree(log_dir)
        raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_exp_path", type=str,
                        help="directory with config files")
    parser.add_argument("log_exp_path", type=str,
                        help="log directory")
    parser.add_argument("--report_path", type=str, default=None,
                        help="report directory")
    parser.add_argument("--idle", action="store_true",
                        help="do not run the experiments")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    config_exp_path = Path(args.config_exp_path).resolve()
    config_paths = [p for p in config_exp_path.iterdir()
                    if p.suffix == '.json']

    log_exp_path = Path(args.log_exp_path).resolve()
    if not args.idle:
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            run(config, log_exp_path, do_plot_frontier=False)

    log_paths = list([p for p in log_exp_path.iterdir() if p.is_dir()])
    log_paths = sorted(log_paths)

    report_path = args.report_path
    if report_path is None:
        report_path = log_exp_path / 'reports'

    report_path.mkdir()
    static_path = report_path / 'static'
    static_path.mkdir()

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
