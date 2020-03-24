from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

LABELS = ['normal', 'outlier']
MSS = [0.1, 2]
MARKERS = ['o', 'x']


def plot_instance_score(ax, scores, target,
                        idx=None,
                        labels=None,
                        mss=None,
                        markers=None,
                        threshold=None,
                        ylim=(None, None),
                        xlabel='Number of Instances',
                        ylabel='Instance Score',
                        ):
    """
    Scatter plot of a batch of outlier or adversarial scores compared to the threshold.

    Parameters
    ----------
    preds
        Dictionary returned by predictions of an outlier or adversarial detector.
    target
        Ground truth.
    labels
        List with names of classification labels.
    threshold
        Threshold used to classify outliers or adversarial instances.
    ylim
        Min and max y-axis values.
    """
    if labels is None:
        labels = LABELS

    if mss is None:
        mss = MSS

    if markers is None:
        markers = MARKERS

    if idx is None:
        idx = np.arange(len(scores))

    df = pd.DataFrame(dict(idx=idx, score=scores, label=target))
    groups = df.groupby('label')
    for name, group in groups:
        ax.plot(group['idx'], group['score'], marker=markers[name], linestyle='',
                ms=mss[name], label=labels[name])

    if threshold is not None:
        ax.plot(df['idx'], np.ones(len(scores)) * threshold, color='g', label='Threshold')

    ax.set_yscale('log')
    if ylim is None:
        ylim = (df['score'].min(), df['score'].max())
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()
    return ax


def plot_feature_outlier_tabular(od_preds: Dict,
                                 X: np.ndarray,
                                 X_recon: np.ndarray = None,
                                 threshold: float = None,
                                 instance_ids: list = None,
                                 max_instances: int = 5,
                                 top_n: int = int(1e12),
                                 outliers_only: bool = False,
                                 feature_names: list = None,
                                 width: float = .2,
                                 figsize: tuple = (20, 10)) -> None:
    """
    Plot feature wise outlier scores for tabular data.

    Parameters
    ----------
    od_preds
        Output of an outlier detector's prediction.
    X
        Batch of instances to apply outlier detection to.
    X_recon
        Reconstructed instances of X.
    threshold
        Threshold used for outlier score to determine outliers.
    instance_ids
        List with indices of instances to display.
    max_instances
        Maximum number of instances to display.
    top_n
        Maixmum number of features to display, ordered by outlier score.
    outliers_only
        Whether to only show outliers or not.
    feature_names
        List with feature names.
    width
        Column width for bar charts.
    figsize
        Tuple for the figure size.
    """
    if outliers_only and instance_ids is None:
        instance_ids = list(np.where(od_preds['data']['is_outlier'])[0])
    elif instance_ids is None:
        instance_ids = list(range(len(od_preds['data']['is_outlier'])))
    n_instances = min(max_instances, len(instance_ids))
    instance_ids = instance_ids[:n_instances]
    n_features = X.shape[1]
    n_cols = 2

    labels_values = ['Original']
    if X_recon is not None:
        labels_values += ['Reconstructed']
    labels_scores = ['Outlier Score']
    if threshold is not None:
        labels_scores = ['Threshold'] + labels_scores

    fig, axes = plt.subplots(nrows=n_instances, ncols=n_cols, figsize=figsize)

    n_subplot = 1
    for i in range(n_instances):

        idx = instance_ids[i]

        fscore = od_preds['data']['feature_score'][idx]
        if top_n >= n_features:
            keep_cols = np.arange(n_features)
        else:
            keep_cols = np.argsort(fscore)[::-1][:top_n]
        fscore = fscore[keep_cols]
        X_idx = X[idx][keep_cols]
        ticks = np.arange(len(keep_cols))

        plt.subplot(n_instances, n_cols, n_subplot)
        if X_recon is not None:
            X_recon_idx = X_recon[idx][keep_cols]
            plt.bar(ticks - width, X_idx, width=width, color='b', align='center')
            plt.bar(ticks, X_recon_idx, width=width, color='g', align='center')
        else:
            plt.bar(ticks, X_idx, width=width, color='b', align='center')
        if feature_names is not None:
            plt.xticks(ticks=ticks, labels=list(np.array(feature_names)[keep_cols]), rotation=45)
        plt.title('Feature Values')
        plt.xlabel('Features')
        plt.ylabel('Feature Values')
        plt.legend(labels_values)
        n_subplot += 1

        plt.subplot(n_instances, n_cols, n_subplot)
        plt.bar(ticks, fscore)
        if threshold is not None:
            plt.plot(np.ones(len(ticks)) * threshold, 'r')
        if feature_names is not None:
            plt.xticks(ticks=ticks, labels=list(np.array(feature_names)[keep_cols]), rotation=45)
        plt.title('Feature Level Outlier Score')
        plt.xlabel('Features')
        plt.ylabel('Outlier Score')
        plt.legend(labels_scores)
        n_subplot += 1

    plt.tight_layout()
    plt.show()



def plot_precision_recall(ax, precisions, recalls, thresholds=None):
    ax.step(recalls, precisions, color='b', alpha=0.2, where='post')
    ax.fill_between(recalls, precisions, alpha=0.2, color='b', step='post')

    if thresholds is not None:
        for p, r, t in zip(precisions, recalls, thresholds):
            ax.annotate('t={0:.2f}'.format(t), xy=(r, p))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall curve')


def plot_f1score(ax, f1scores, thresholds):
    ax.step(thresholds, f1scores, color='b', alpha=0.2, where='post')
    ax.fill_between(thresholds, f1scores, alpha=0.2, color='b', step='post')

    for f1, t in zip(f1scores, thresholds):
        ax.annotate('t={0:.2f}'.format(t), xy=(t, f1))

    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0.0, 1.05])
    ax.set_title('F1 Score')


def plot_data_2d(ax, x_normal, x_anomaly):
    ax.scatter(x_normal[:, 0], x_normal[:, 1], c='white', s=20, edgecolor='k')
    ax.scatter(x_anomaly[:, 0], x_anomaly[:, 1], c='red', s=20, edgecolor='k')


def plot_frontier(ax, xx, yy, Z):
    upper = min(0, Z.max())
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), upper, 7), cmap=plt.cm.PuBu)
    if Z.max() >= 0:
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        ax.contourf(xx, yy, Z, levels=[0, Z.max() + 1e-3], colors='palevioletred')
