
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

    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    y_true = y_true[sort_idx]

    n = y_true.shape[0]

    precisions = []
    recalls = []
    f1scores = []

    for p in thresholds:
        n_anom = floor(p*0.01*n)
        thresh = scores[-n_anom]
        y_pred = (scores >= thresh).astype(np.int)
        precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1scores.append(float(f1score))

    prf1 = dict(
        precisions=precisions,
        recalls=recalls,
        f1scores=f1scores,
        thresholds=thresholds
    )

    return prf1


