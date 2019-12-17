
from math import floor

import numpy as np

from sklearn.metrics import precision_recall_fscore_support


def select_threshold(thresholds, metric_scores):
    best_i = np.argmax(metric_scores)
    return thresholds[best_i]


def precision_recall_curve_scores(y_true, score, threshold_percs):
    """
    :param y_true: ground truth labels
    :param scores: outlier scores (the higher the more anomalous)
    :param threshold_percs: percentage of X considered to be normal based on the outlier score.
    :return:
    """

    precisions = []
    recalls = []
    f1scores = []
    thresholds = []

    # Look at first p frac anomalies
    for threshold_perc in threshold_percs:
        thresh = np.percentile(score, threshold_perc)
        y_pred = (score > thresh).astype(int)
        precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1scores.append(float(f1score))
        thresholds.append(float(thresh))

    if isinstance(threshold_percs, np.ndarray):
        threshold_percs = threshold_percs.tolist()

    prf1 = dict(
        precisions=precisions,
        recalls=recalls,
        f1scores=f1scores,
        thresholds=thresholds,
        threshold_percs=threshold_percs
    )

    return prf1


def get_frontier(od, x):
    lower, upper = x.min(axis=0), x.max(axis=0)
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 500),
                         np.linspace(lower[1], upper[1], 500))
    Z = od.predict(np.c_[xx.ravel(), yy.ravel()])['data']['instance_score']
    Z -= od.threshold
    Z = Z.reshape(xx.shape)
    return xx, yy, Z
