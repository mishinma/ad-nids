
from math import floor

import numpy as np

from sklearn.metrics import precision_recall_fscore_support


DEFAULT_THRESHOLDS = [1, 2, 5, 10, 15, 20, 30]


def select_threshold(thresholds, metric_scores):
    best_i = np.argmax(metric_scores)
    return thresholds[best_i]


def precision_recall_curve_scores(y_true, scores, thresholds=None):
    """
    :param y_true: ground truth labels
    :param scores: anomaly scores (the higher the more anomalous)
    :param thresholds: to select p% most anomalous samples
    :return:
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    y_true = np.array(y_true)
    scores = np.array(scores)

    sort_idx = np.argsort(scores)[::-1]  # Reverse; the lower the more abnormal
    scores = scores[sort_idx]
    y_true = y_true[sort_idx]

    n = y_true.shape[0]

    precisions = []
    recalls = []
    f1scores = []

    thresholds_val = []

    # Look at first p frac anomalies
    for p in thresholds:
        n_anom = floor(p*0.01*n)
        thresh = scores[-n_anom]
        y_pred = (scores < thresh).astype(np.int)
        precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1scores.append(float(f1score))
        thresholds_val.append(float(thresh))

    prf1 = dict(
        precisions=precisions,
        recalls=recalls,
        f1scores=f1scores,
        thresholds=thresholds_val
    )

    return prf1


def get_frontier(model, x):
    lower, upper = x.min(axis=0), x.max(axis=0)
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 500),
                         np.linspace(lower[1], upper[1], 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

