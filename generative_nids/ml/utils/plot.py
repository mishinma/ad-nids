
import numpy as np
import matplotlib.pyplot as plt


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


def plot_data_2d(ax, x_normal, x_anomaly):
    ax.scatter(x_normal[:, 0], x_normal[:, 1], c='white', s=20, edgecolor='k')
    ax.scatter(x_anomaly[:, 0], x_anomaly[:, 1], c='red', s=20, edgecolor='k')


def plot_frontier(ax, xx, yy, Z):
    upper = min(0, Z.max())
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), upper, 7), cmap=plt.cm.PuBu)
    if Z.max() >= 0:
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        ax.contourf(xx, yy, Z, levels=[0, Z.max() + 1e-3], colors='palevioletred')


