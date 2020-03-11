import numpy as np
import pandas as pd


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