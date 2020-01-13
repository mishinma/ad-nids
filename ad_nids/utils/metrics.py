
from math import floor

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.metrics import precision_recall_fscore_support


def concatenate_preds(preds, other_preds):
    if preds['data'].get('feature_score') is not None:
        preds['data']['feature_score'] = np.concatenate(
            [preds['data']['feature_score'],  other_preds['data']['feature_score']]
        )
    preds['data']['instance_score'] = np.concatenate(
        [preds['data']['instance_score'], other_preds['data']['instance_score']]
    )
    preds['data']['is_outlier'] = np.concatenate(
        [preds['data']['is_outlier'], other_preds['data']['is_outlier']]
    )
    return preds


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

    prf1 = dict(
        precisions=np.array(precisions),
        recalls=np.array(recalls),
        f1scores=np.array(f1scores),
        thresholds=np.array(thresholds),
        threshold_percs=np.array(threshold_percs)
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


def cov_elbo_type(cov_elbo, X):
    cov_elbo_type, cov = [*cov_elbo][0], [*cov_elbo.values()][0]
    if cov_elbo_type in ['cov_full', 'cov_diag']:
        cov = tfp.stats.covariance(X.reshape(X.shape[0], -1))
        if cov_elbo_type == 'cov_diag':  # infer standard deviation from covariance matrix
            cov = tf.math.sqrt(tf.linalg.diag_part(cov))

    return {cov_elbo_type: tf.dtypes.cast(cov, tf.float32)}

