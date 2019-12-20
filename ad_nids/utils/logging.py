import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from alibi_detect.utils.saving import save_detector
from alibi_detect.utils.visualize import plot_instance_score
from ad_nids.utils.metrics import get_frontier
from ad_nids.utils.misc import jsonify
from ad_nids.utils.plot import plot_precision_recall, \
    plot_f1score, plot_data_2d, plot_frontier


def get_log_dir(log_root_dir, config_name):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + config_name
    log_dir = Path(log_root_dir) / uniq_name
    log_dir = log_dir.resolve()
    return log_dir


def log_experiment(log_dir, config, dataset_meta, detector,
                   eval_results):

    with open(log_dir/'config.json', 'w') as f:
        json.dump(config, f)

    with open(log_dir/'dataset_meta.json', 'w') as f:
        json.dump(dataset_meta, f)

    save_detector(detector, str(log_dir/'detector'))

    with open(log_dir/'eval_results.json', 'w') as f:
        json.dump(jsonify(eval_results), f)


def log_plot_prf1_curve(log_dir, prf1_curve):
    # plot train precision recall curve
    fig, ax = plt.subplots(1, 1)
    plot_precision_recall(
        ax,
        prf1_curve['precisions'],
        prf1_curve['recalls'],
        prf1_curve['threshold_percs']
    )
    fig.savefig(log_dir / 'train_pr_curve.png')
    plt.close()

    # plot train f1 score curve
    fig, ax = plt.subplots(1, 1)
    plot_f1score(
        ax,
        prf1_curve['f1scores'], prf1_curve['threshold_percs']
    )
    fig.savefig(log_dir / 'train_f1_curve.png')
    plt.close()


def log_plot_frontier(log_dir, detector,
                      X_train, y_train, X_test, y_test):

    fig, ax = plt.subplots(1, 1)
    xx, yy, Z = get_frontier(detector, X_train)
    plot_frontier(ax, xx, yy, -Z)
    X_train_norm = X_train[~y_train.astype(bool)]
    X_train_anom = X_train[y_train.astype(bool)]
    plot_data_2d(ax, X_train_norm, X_train_anom)
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


def log_plot_instance_score(log_dir, detector_preds, y_true,
                            detector_threshold, train=False, labels=None):

    if labels is None:
        labels = ['normal', 'outlier']

    plot_instance_score(detector_preds, y_true, labels, detector_threshold)
    set_ = 'train_' if train else 'test_'
    filename = set_ + 'instance_score.png'
    plt.savefig(log_dir / filename)
    plt.close()
